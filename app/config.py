import os
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    APP_NAME: str = "Clinical Audio Assessment Pipeline"
    DEBUG: bool = False
    
    # MongoDB settings
    MONGODB_URI: str = "mongodb://localhost:27017"
    MONGODB_DB_NAME: str = "clinical_assessments_db"
    MONGODB_COLLECTION: str = "assessments"

    # LLM Settings
    OPENAI_API_KEY: Optional[str] = None
    LLM_MODEL: str = "gpt-4o-mini"
    LLM_TEMPERATURE: float = 0.0

    # Whisper Settings (options: "api", "local", "mock")
    WHISPER_MODE: str = "local"
    WHISPER_MODEL_SIZE: str = "base"  # tiny, base, small, medium, large

    # Extraction confidence threshold (0.0 to 1.0)
    MIN_CONFIDENCE_THRESHOLD: float = 0.50


settings = Settings()
