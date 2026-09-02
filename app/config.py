"""Application configuration using pydantic-settings."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Global application settings loaded from environment variables or .env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Application settings
    APP_NAME: str = "Clinical-Assessment-Pipeline"
    ENV: str = "development"
    DEBUG: bool = True
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # OpenAI & LLM settings
    OPENAI_API_KEY: str = ""
    WHISPER_MODEL: str = "whisper-1"
    EXTRACTION_MODEL: str = "gpt-4o"
    CONFIDENCE_THRESHOLD: float = 0.75

    # MongoDB settings
    MONGO_URI: str = "mongodb://localhost:27017"
    MONGO_DB_NAME: str = "clinical_db"
    MONGO_COLLECTION: str = "assessments"


settings = Settings()
