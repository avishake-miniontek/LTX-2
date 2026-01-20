import os
import shutil
import tempfile
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from app.cleanup_service import cleanup_service
from app.config import settings
from app.logger import log
from app.models import JobStatus, init_db
from app.pipeline_manager import pipeline_manager
from app.queue_manager import queue_manager
from app.schemas import (
    HealthResponse,
    JobStatusResponse,
    VideoGenerationRequest,
    VideoGenerationResponse,
)


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ANN201, ARG001
    """Lifecycle manager for startup and shutdown events."""

    # Startup
    log.info("Starting LTX Video Generation Server...")

    try:
        # Initialize database
        await init_db()
        log.info("Database initialized")

        # Create output directory
        os.makedirs(settings.output_dir, exist_ok=True)  # noqa: PTH103
        log.info(f"Output directory: {settings.output_dir}")

        # Run startup cleanup
        await cleanup_service.cleanup_on_startup()

        # Initialize pipeline
        pipeline_manager.initialize()

        # Start queue processor
        queue_manager.start()

        # Start cleanup service
        cleanup_service.start()

        log.info("Server startup complete")

    except Exception as e:
        log.error(f"Startup failed: {e}")
        raise

    yield

    # Shutdown
    log.info("Shutting down server...")

    try:
        queue_manager.stop()
        cleanup_service.stop()
        pipeline_manager.cleanup()
        log.info("Server shutdown complete")

    except Exception as e:
        log.error(f"Shutdown error: {e}")


app = FastAPI(
    title="LTX Video Generation API",
    description="Production-ready API for LTX-2 video generation",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health_check():  # noqa: ANN201
    """Health check endpoint."""

    try:
        queue_size = await queue_manager.get_queue_size()
        processing = queue_manager.is_processing()

        return HealthResponse(
            status="healthy",
            timestamp=datetime.now(),  # noqa: DTZ005
            queue_size=queue_size,
            processing=processing,
        )
    except Exception as e:
        log.error(f"Health check failed: {e}")
        raise HTTPException(  # noqa: B904
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service unhealthy"
        )


@app.post("/ltx-2-generate-video", response_model=VideoGenerationResponse)
async def generate_video(  # noqa: ANN201
    prompt: str = Form(...),
    negative_prompt: Optional[str] = Form(None),
    seed: Optional[int] = Form(None),
    width: int = Form(1920),
    height: int = Form(1024),
    fps: int = Form(24),
    duration: float = Form(10.0),
    num_inference_steps: int = Form(20),
    cfg_guidance_scale: float = Form(3.0),
    image: Optional[UploadFile] = File(None),  # noqa: B008
):
    """
    Generate a video from text prompt and optional image.

    This endpoint queues video generation jobs and returns immediately with a job ID.
    Use the /job/{job_id} endpoint to check status and download the video.
    """

    try:
        # Validate request
        request = VideoGenerationRequest(
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            width=width,
            height=height,
            fps=fps,
            duration=duration,
            num_inference_steps=num_inference_steps,
            cfg_guidance_scale=cfg_guidance_scale,
        )

        # Handle optional image upload
        image_path = None
        if image:
            try:
                # Validate file type
                if not image.content_type.startswith('image/'):
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="File must be an image"
                    )

                # Save uploaded image temporarily
                temp_dir = tempfile.mkdtemp()
                image_path = os.path.join(temp_dir, f"input_{image.filename}")  # noqa: PTH118

                with open(image_path, "wb") as buffer:
                    shutil.copyfileobj(image.file, buffer)

                log.info(f"Image uploaded: {image_path}")

            except Exception as e:
                log.error(f"Failed to process image: {e}")
                raise HTTPException(  # noqa: B904
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Failed to process image: {str(e)}"  # noqa: RUF010
                )

        # Add job to queue
        job_id = await queue_manager.add_job(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            seed=request.seed,
            width=request.width,
            height=request.height,
            fps=request.fps,
            duration=request.duration,
            num_inference_steps=request.num_inference_steps,
            cfg_guidance_scale=request.cfg_guidance_scale,
            image_path=image_path,
        )

        return VideoGenerationResponse(
            job_id=job_id,
            status="queued",
            message="Video generation job queued successfully. Use /job/{job_id} to check status."
        )

    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Failed to create video generation job: {e}")
        raise HTTPException(  # noqa: B904
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create job: {str(e)}"  # noqa: RUF010
        )


@app.get("/job/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):  # noqa: ANN201
    """Get the status of a video generation job."""

    try:
        job = await queue_manager.get_job_status(job_id)

        if job is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Job not found"
            )

        return JobStatusResponse.model_validate(job)

    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Failed to get job status: {e}")
        raise HTTPException(  # noqa: B904
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get job status: {str(e)}"  # noqa: RUF010
        )


@app.get("/job/{job_id}/download")
async def download_video(job_id: str):  # noqa: ANN201
    """Download the generated video file."""

    try:
        job = await queue_manager.get_job_status(job_id)

        if job is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Job not found"
            )

        if job.status != JobStatus.COMPLETED:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Video not ready. Current status: {job.status}"
            )

        if not job.output_path or not os.path.exists(job.output_path):  # noqa: PTH110
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Video file not found"
            )

        return FileResponse(
            path=job.output_path,
            media_type="video/mp4",
            filename=f"video_{job_id}.mp4"
        )

    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Failed to download video: {e}")
        raise HTTPException(  # noqa: B904
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to download video: {str(e)}"  # noqa: RUF010
        )


@app.get("/queue")
async def get_queue_info():  # noqa: ANN201
    """Get information about the current queue."""

    try:
        queue_size = await queue_manager.get_queue_size()
        processing = queue_manager.is_processing()

        return {
            "queue_size": queue_size,
            "processing": processing,
            "message": f"{queue_size} jobs in queue, processing: {processing}"
        }

    except Exception as e:
        log.error(f"Failed to get queue info: {e}")
        raise HTTPException(  # noqa: B904
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get queue info: {str(e)}"  # noqa: RUF010
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=False,
        log_level=settings.log_level.lower(),
        timeout_keep_alive=settings.request_timeout,
    )
