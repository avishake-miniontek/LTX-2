import asyncio
import os
from datetime import datetime, timedelta
from pathlib import Path

from sqlalchemy import delete, select

from app import models
from app.config import settings
from app.logger import log
from app.models import JobStatus, VideoJob


class CleanupService:
    """Service for cleaning up old video files."""

    _instance = None
    _task = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, '_initialized'):
            self._task = None
            self._initialized = True

    async def cleanup_on_startup(self):  # noqa: ANN201
        """Perform cleanup on server startup."""

        log.info("Running startup cleanup...")

        try:
            # Clean up orphaned files in output directory
            await self._cleanup_orphaned_files()

            # Reset any jobs that were processing when server crashed
            await self._reset_processing_jobs()

            log.info("Startup cleanup completed")

        except Exception as e:
            log.error(f"Startup cleanup failed: {e}")

    def start(self):  # noqa: ANN201
        """Start the periodic cleanup task."""

        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._periodic_cleanup())
            log.info("Cleanup service started")

    def stop(self):  # noqa: ANN201
        """Stop the cleanup task."""

        if self._task and not self._task.done():
            self._task.cancel()
            log.info("Cleanup service stopped")

    async def _periodic_cleanup(self):  # noqa: ANN202
        """Periodically clean up old files."""

        interval = settings.cleanup_interval_minutes * 60  # Convert to seconds

        while True:
            try:
                await asyncio.sleep(interval)
                log.info("Running periodic cleanup...")
                await self._cleanup_old_files()

            except asyncio.CancelledError:
                log.info("Cleanup service cancelled")
                break
            except Exception as e:
                log.error(f"Periodic cleanup error: {e}")

    async def _cleanup_old_files(self):  # noqa: ANN202
        """Remove video files older than retention period."""

        try:
            retention_delta = timedelta(hours=settings.file_retention_hours)
            cutoff_time = datetime.now() - retention_delta  # noqa: DTZ005

            # Get old completed jobs
            async with models.AsyncSessionLocal() as session:
                result = await session.execute(
                    select(VideoJob)
                    .where(VideoJob.status == JobStatus.COMPLETED)
                    .where(VideoJob.completed_at < cutoff_time)
                )
                old_jobs = result.scalars().all()

                deleted_count = 0
                for job in old_jobs:
                    if job.output_path and os.path.exists(job.output_path):  # noqa: PTH110
                        try:
                            os.remove(job.output_path)  # noqa: PTH107
                            log.info(f"Deleted old video file: {job.output_path}")
                            deleted_count += 1
                        except Exception as e:
                            log.error(f"Failed to delete {job.output_path}: {e}")

                # Delete job records
                if old_jobs:
                    await session.execute(
                        delete(VideoJob)
                        .where(VideoJob.status == JobStatus.COMPLETED)
                        .where(VideoJob.completed_at < cutoff_time)
                    )
                    await session.commit()
                    log.info(f"Cleaned up {deleted_count} old video files and {len(old_jobs)} job records")

        except Exception as e:
            log.error(f"Cleanup old files failed: {e}")

    async def _cleanup_orphaned_files(self):  # noqa: ANN202
        """Remove files in output directory that don't have corresponding job records."""

        try:
            if not os.path.exists(settings.output_dir):  # noqa: PTH110
                return

            # Get all output files
            output_files = set()
            for file in Path(settings.output_dir).glob("*.mp4"):
                output_files.add(str(file))

            # Get all valid output paths from database
            async with models.AsyncSessionLocal() as session:
                result = await session.execute(
                    select(VideoJob.output_path).where(VideoJob.output_path.isnot(None))
                )
                valid_paths = set(row[0] for row in result.all())  # noqa: C401

            # Delete orphaned files
            orphaned_files = output_files - valid_paths
            deleted_count = 0

            for filepath in orphaned_files:
                try:
                    if os.path.exists(filepath):  # noqa: PTH110
                        os.remove(filepath)  # noqa: PTH107
                        log.info(f"Deleted orphaned file: {filepath}")
                        deleted_count += 1
                except Exception as e:
                    log.error(f"Failed to delete orphaned file {filepath}: {e}")

            if deleted_count > 0:
                log.info(f"Cleaned up {deleted_count} orphaned files")

        except Exception as e:
            log.error(f"Cleanup orphaned files failed: {e}")

    async def _reset_processing_jobs(self):  # noqa: ANN202
        """Reset jobs that were stuck in processing state."""

        try:
            async with models.AsyncSessionLocal() as session:
                result = await session.execute(
                    select(VideoJob).where(VideoJob.status == JobStatus.PROCESSING)
                )
                stuck_jobs = result.scalars().all()

                if stuck_jobs:
                    for job in stuck_jobs:
                        job.status = JobStatus.FAILED
                        job.error_message = "Server restart during processing"
                        job.completed_at = datetime.now()  # noqa: DTZ005

                    await session.commit()
                    log.info(f"Reset {len(stuck_jobs)} stuck jobs")

        except Exception as e:
            log.error(f"Reset processing jobs failed: {e}")


# Global instance
cleanup_service = CleanupService()
