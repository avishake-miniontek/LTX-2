import asyncio
import os
import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import select, update

from app import models
from app.config import settings
from app.logger import log
from app.models import JobStatus, VideoJob
from app.pipeline_manager import pipeline_manager


class QueueManager:
    """Manages the video generation queue."""

    _instance = None
    _processing = False
    _task = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, '_initialized'):
            self._processing = False
            self._task = None
            self._initialized = True

    def start(self):  # noqa: ANN201
        """Start the queue processor."""
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._process_queue())
            log.info("Queue processor started")

    def stop(self):  # noqa: ANN201
        """Stop the queue processor."""
        if self._task and not self._task.done():
            self._task.cancel()
            log.info("Queue processor stopped")

    async def add_job(
        self,
        prompt: str,
        negative_prompt: Optional[str],
        seed: Optional[int],
        width: int,
        height: int,
        fps: int,
        duration: float,
        num_inference_steps: int,
        cfg_guidance_scale: float,
        image_path: Optional[str] = None,
    ) -> str:
        """Add a new job to the queue."""

        job_id = str(uuid.uuid4())

        async with models.AsyncSessionLocal() as session:
            job = VideoJob(
                id=job_id,
                status=JobStatus.QUEUED,
                prompt=prompt,
                negative_prompt=negative_prompt,
                seed=seed,
                width=width,
                height=height,
                fps=fps,
                duration=duration,
                num_inference_steps=num_inference_steps,
                cfg_guidance_scale=cfg_guidance_scale,
                input_image_path=image_path,
            )
            session.add(job)
            await session.commit()

        log.info(f"Job {job_id} added to queue" + (f" with image: {image_path}" if image_path else ""))
        return job_id

    async def get_job_status(self, job_id: str) -> Optional[VideoJob]:
        """Get job status from database."""

        async with models.AsyncSessionLocal() as session:
            result = await session.execute(
                select(VideoJob).where(VideoJob.id == job_id)
            )
            return result.scalar_one_or_none()

    async def get_queue_size(self) -> int:
        """Get number of queued jobs."""

        async with models.AsyncSessionLocal() as session:
            result = await session.execute(
                select(VideoJob).where(
                    VideoJob.status.in_([JobStatus.QUEUED, JobStatus.PROCESSING])
                )
            )
            return len(result.scalars().all())

    def is_processing(self) -> bool:
        """Check if a job is currently being processed."""
        return self._processing

    async def _process_queue(self):  # noqa: ANN202
        """Background task to process queued jobs."""

        log.info("Queue processor running")

        while True:
            try:
                await asyncio.sleep(2)  # Check every 2 seconds

                if self._processing:
                    continue

                # Get next queued job
                async with models.AsyncSessionLocal() as session:
                    result = await session.execute(
                        select(VideoJob)
                        .where(VideoJob.status == JobStatus.QUEUED)
                        .order_by(VideoJob.created_at)
                        .limit(1)
                    )
                    job = result.scalar_one_or_none()

                    if job is None:
                        continue

                    # Mark as processing
                    self._processing = True
                    await session.execute(
                        update(VideoJob)
                        .where(VideoJob.id == job.id)
                        .where(VideoJob.status == JobStatus.QUEUED)
                        .values(
                            status=JobStatus.PROCESSING,
                            started_at=datetime.now()  # noqa: DTZ005
                        )
                    )
                    await session.commit()

                    job_id = job.id

                log.info(f"Processing job {job_id}")

                # Process the job
                await self._process_job(job_id)

            except asyncio.CancelledError:
                log.info("Queue processor cancelled")
                break
            except Exception as e:
                log.error(f"Error in queue processor: {e}")
                self._processing = False

    async def _process_job(self, job_id: str):  # noqa: ANN202
        """Process a single job."""

        try:
            # Get job details
            async with models.AsyncSessionLocal() as session:
                result = await session.execute(
                    select(VideoJob).where(VideoJob.id == job_id)
                )
                job = result.scalar_one_or_none()

                if job is None:
                    log.error(f"Job {job_id} not found")
                    self._processing = False
                    return

            # Generate output path
            output_filename = f"{job_id}.mp4"
            output_path = os.path.join(settings.output_dir, output_filename)  # noqa: PTH118
            os.makedirs(settings.output_dir, exist_ok=True)  # noqa: PTH103

            # Log image info
            if job.input_image_path:
                if os.path.exists(job.input_image_path):
                    log.info(f"Using input image: {job.input_image_path}")
                else:
                    log.warning(f"Input image not found: {job.input_image_path}")

            # Generate video
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                None,
                pipeline_manager.generate_video,
                job.prompt,
                job.negative_prompt,
                job.seed,
                job.width,
                job.height,
                job.fps,
                job.duration,
                job.num_inference_steps,
                job.cfg_guidance_scale,
                output_path,
                job.input_image_path,
            )

            # Update job as completed
            async with models.AsyncSessionLocal() as session:
                await session.execute(
                    update(VideoJob)
                    .where(VideoJob.id == job_id)
                    .where(VideoJob.status == JobStatus.PROCESSING)
                    .values(
                        status=JobStatus.COMPLETED,
                        output_path=output_path,
                        completed_at=datetime.now()  # noqa: DTZ005
                    )
                )
                await session.commit()


            log.info(f"Job {job_id} completed successfully")

        except Exception as e:
            log.error(f"Job {job_id} failed: {e}")

            # Update job as failed
            async with models.AsyncSessionLocal() as session:
                await session.execute(
                    update(VideoJob)
                    .where(VideoJob.id == job_id)
                    .where(VideoJob.status == JobStatus.PROCESSING)
                    .values(
                        status=JobStatus.FAILED,
                        error_message=str(e),
                        completed_at=datetime.now()  # noqa: DTZ005
                    )
                )
                await session.commit()

        finally:
            self._processing = False


# Global instance
queue_manager = QueueManager()
