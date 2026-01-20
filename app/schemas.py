from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, field_validator

from ltx_pipelines.utils.constants import DEFAULT_NEGATIVE_PROMPT


class VideoGenerationRequest(BaseModel):
    """Request model for video generation."""

    prompt: str = Field(..., description="Text prompt for video generation", min_length=1)
    negative_prompt: Optional[str] = Field(default=DEFAULT_NEGATIVE_PROMPT, description="Negative prompt to avoid certain features")
    seed: Optional[int] = Field(None, description="Random seed for reproducibility", ge=0)
    width: int = Field(1920, description="Video width in pixels", ge=128, le=3840)
    height: int = Field(1024, description="Video height in pixels", ge=128, le=2160)
    fps: int = Field(24, description="Frames per second", ge=1, le=60)
    duration: float = Field(10.0, description="Video duration in seconds", ge=0.1, le=60.0)
    num_inference_steps: int = Field(20, description="Number of inference steps", ge=1, le=100)
    cfg_guidance_scale: float = Field(3.0, description="Classifier-free guidance scale", ge=1.0, le=20.0)

    @field_validator('duration')
    @classmethod
    def validate_duration(cls, v):  # noqa: ANN001, ANN206
        if v <= 0:
            raise ValueError("Duration must be positive")
        return v


class VideoGenerationResponse(BaseModel):
    """Response model for video generation request."""

    job_id: str = Field(..., description="Unique job identifier")
    status: str = Field(..., description="Current job status")
    message: str = Field(..., description="Response message")


class JobStatusResponse(BaseModel):
    """Response model for job status check."""

    job_id: str = Field(alias="id")
    status: str
    prompt: str
    output_path: Optional[str] = None
    error_message: Optional[str] = None
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    model_config = {
        "from_attributes": True,
        "populate_by_name": True,
    }


class HealthResponse(BaseModel):
    """Response model for health check."""

    status: str = Field(..., description="Health status")
    timestamp: datetime = Field(..., description="Current server time")
    queue_size: int = Field(..., description="Number of jobs in queue")
    processing: bool = Field(..., description="Whether a job is currently processing")
