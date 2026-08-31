from functools import lru_cache
from typing import Literal, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    google_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None

    whisper_backend: Literal["local", "openai"] = "local"
    whisper_model: str = "small"
    whisper_language: Optional[str] = "en"

    llm_model: str = "gemini-2.5-flash"
    extraction_confidence_threshold: float = 0.6

    mongodb_uri: str = "mongodb://localhost:27017"
    mongodb_database: str = "clinical"
    mongodb_collection: str = "assessments"

    max_upload_bytes: int = 50 * 1024 * 1024
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()
