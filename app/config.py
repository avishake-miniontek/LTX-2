from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Server Configuration
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"
    log_file: str = "logs/ltx_server.log"

    # Model Paths
    model_checkpoint: str = "models/ltx-2-19b-dev-fp8.safetensors"
    distilled_lora: str = "models/ltx-2-19b-distilled-lora-384.safetensors"
    spatial_upsampler: str = "models/ltx-2-spatial-upscaler-x2-1.0.safetensors"
    gemma_root: str = "models/gemma-3-12b-it"

    # Generation Settings
    output_dir: str = "./output"
    database_url: str = "sqlite+aiosqlite:///./ltx_video.db"

    # Cleanup Settings
    file_retention_hours: int = 4
    cleanup_interval_minutes: int = 30

    # Request Timeout
    request_timeout: int = 3600

    # Device
    device: str = "cuda"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )


settings = Settings()
