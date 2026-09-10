from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Database
    MONGODB_URL: str = "mongodb://localhost:27017"
    DATABASE_NAME: str = "stance_health"
    COLLECTION_NAME: str = "assessments"

    # Extraction & Confidence Thresholds
    CONFIDENCE_THRESHOLD: float = 0.70
    WHISPER_MODEL: str = "base"

    # LLM Settings
    LLM_PROVIDER: str = "gemini"  # gemini, openai, anthropic, ollama
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_MODEL: str = "gpt-4o-mini"
    
    GEMINI_API_KEY: Optional[str] = None
    GEMINI_MODEL: str = "gemini-1.5-flash"
    
    ANTHROPIC_API_KEY: Optional[str] = None
    ANTHROPIC_MODEL: str = "claude-3-haiku-20240307"

    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3"

    # Server
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000


settings = Settings()
