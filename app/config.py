from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = (
        "Stance Health Clinical Assessment API"
    )

    app_version: str = "1.0.0"

    mongodb_uri: str = (
        "mongodb://localhost:27017"
    )

    mongodb_database: str = (
        "stance_assessment"
    )

    ollama_base_url: str = (
        "http://localhost:11434"
    )

    ollama_model: str = (
        "llama3.2:3b"
    )

    whisper_model: str = "base"

    whisper_language: str = "en"

    confidence_threshold: float = 0.70

    max_audio_size_mb: int = 100


@lru_cache
def get_settings() -> Settings:
    return Settings()
