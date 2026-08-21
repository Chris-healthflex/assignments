from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    mongodb_uri: str = "mongodb://localhost:27017"
    mongodb_database: str = "clinical_assessments"
    mongodb_collection: str = "assessments"
    whisper_model: str = "base"
    groq_api_key: str | None = None
    extraction_model: str = "gpt-4o-mini"
    confidence_threshold: float = 0.75

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()
