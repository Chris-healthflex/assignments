"""Application configuration, loaded from environment / .env."""
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # --- LLM ---
    # Provider is swappable: "openai" | "google" | "groq"
    llm_provider: str = "openai"
    llm_model: str = ""
    llm_temperature: float = 0.0

    openai_api_key: str = ""
    google_api_key: str = ""
    groq_api_key: str = ""
    local_llm_base_url: str = "http://localhost:11434/v1"

    # --- Whisper ---
    whisper_model: str = "base.en"
    whisper_device: str = "cpu"
    whisper_compute_type: str = "int8"

    # --- MongoDB ---
    mongodb_uri: str = "mongodb://localhost:27017"
    database_name: str = "stance_health"
    assessments_collection: str = "assessments"

    # --- Extraction quality gate ---
    confidence_threshold: float = 0.70

    # --- Upload guards ---
    max_upload_bytes: int = 50 * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    return Settings()
