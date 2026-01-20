import os
import sys

from loguru import logger

from app.config import settings


def setup_logger():  # noqa: ANN201
    """Configure loguru logger with file and console output."""

    # Remove default handler
    logger.remove()

    # Create logs directory if it doesn't exist
    log_dir = os.path.dirname(settings.log_file)  # noqa: PTH120
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)  # noqa: PTH103

    # Add console handler
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",  # noqa: E501
        level=settings.log_level,
        colorize=True,
    )

    # Add file handler with rotation
    logger.add(
        settings.log_file,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level=settings.log_level,
        rotation="100 MB",
        retention="30 days",
        compression="zip",
    )

    logger.info("Logger initialized successfully")
    return logger


log = setup_logger()
