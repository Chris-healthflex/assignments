from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    # App
    APP_NAME: str = "Clinical Assessment Service"
    API_V1_PREFIX: str = "/api/v1"

    # MongoDB
    MONGO_URI: str = "mongodb://localhost:27017"
    MONGO_DB_NAME: str = "clinical_assessments"

    # Whisper
    WHISPER_MODEL: str = "small"
    WHISPER_DEVICE: str = "cpu"
    WHISPER_COMPUTE_TYPE: str = "int8"

    # Groq LLM
    GROQ_API_KEY: str = ""
    GROQ_MODEL: str = "llama-3.3-70b-versatile"
    LLM_TEMPERATURE: float = 0.0

    # Confidence
    CONFIDENCE_THRESHOLD: float = 0.75
    FUZZY_SOURCE_MATCH_THRESHOLD: int = 85
    FUZZY_PARTIAL_MATCH_THRESHOLD: int = 70

    # Audio limits
    MAX_UPLOAD_SIZE_MB: int = 50
    MAX_AUDIO_DURATION_SEC: int = 1800

    # Retry
    MAX_RETRIES: int = 1

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

@lru_cache()
def get_settings() -> Settings:
    return Settings()

settings = get_settings()