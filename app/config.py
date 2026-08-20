"""Application settings, loaded from environment / .env."""

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # App
    app_name: str = "v2n-first-assessment"
    env: Literal["local", "dev", "prod"] = "local"
    log_level: str = "INFO"

    # MongoDB
    mongodb_uri: str = "mongodb://localhost:27017"
    mongodb_db: str = "v2n"
    mongodb_assessments_collection: str = "first_assessments"

    # Whisper
    whisper_backend: Literal["local", "api"] = "local"
    whisper_model: str = "base"
    whisper_device: str = "cpu"
    whisper_language: str | None = "en"

    # Extraction agent
    anthropic_api_key: str | None = None
    extraction_model: str = "claude-opus-5"
    extraction_max_retries: int = 2


@lru_cache
def get_settings() -> Settings:
    """Cached accessor so settings are parsed once per process."""
    return Settings()
