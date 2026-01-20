from datetime import datetime
from enum import Enum

from sqlalchemy import Column, DateTime, Float, Integer, String
from sqlalchemy import Enum as SQLEnum
from sqlalchemy.ext.asyncio import AsyncAttrs, AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from app.config import settings


class Base(AsyncAttrs, DeclarativeBase):
    """Base class for all database models."""
    pass  # noqa: PIE790


class JobStatus(str, Enum):
    """Status of video generation job."""
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class VideoJob(Base):
    """Model for tracking video generation jobs."""
    __tablename__ = "video_jobs"

    id = Column(String, primary_key=True, index=True)
    status = Column(SQLEnum(JobStatus), default=JobStatus.QUEUED, index=True)
    prompt = Column(String, nullable=False)
    negative_prompt = Column(String, nullable=True)
    seed = Column(Integer, nullable=True)
    width = Column(Integer, nullable=False)
    height = Column(Integer, nullable=False)
    fps = Column(Integer, nullable=False)
    duration = Column(Float, nullable=False)
    num_inference_steps = Column(Integer, nullable=False)
    cfg_guidance_scale = Column(Float, nullable=False)
    input_image_path = Column(String, nullable=True)
    output_path = Column(String, nullable=True)
    error_message = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.now, index=True)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)


# Create async engine and session
engine = create_async_engine(
    settings.database_url,
    echo=False,
    future=True,
)

AsyncSessionLocal = async_sessionmaker(
    engine,
    expire_on_commit=False,
    class_=AsyncSession,
)


async def init_db():  # noqa: ANN201
    """Initialize database tables."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
