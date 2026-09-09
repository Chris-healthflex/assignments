from __future__ import annotations

import os
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    APP_NAME: str = "Stance Health Clinical Assessment API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False

    # MongoDB Settings
    MONGODB_URI: str = "mongodb://localhost:27017"
    MONGODB_DATABASE: str = "stance_assessment"

    # Whisper Settings
    WHISPER_BACKEND: str = "local"  # "local" or "api"
    WHISPER_MODEL: str = "base"     # tiny, base, small, medium, large
    WHISPER_LANGUAGE: str = "en"
    MAX_AUDIO_SIZE_MB: int = 100

    # Extraction / Guardrails
    CONFIDENCE_THRESHOLD: float = 0.70

    # LLM Settings
    LLM_PROVIDER: str = "auto"      # "google", "openai", "anthropic", "ollama", "mock"
    LLM_MODEL: Optional[str] = None
    OPENAI_API_KEY: Optional[str] = None
    GOOGLE_API_KEY: Optional[str] = None
    ANTHROPIC_API_KEY: Optional[str] = None
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3.2:3b"


settings = Settings()
