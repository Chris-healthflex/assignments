import json
from functools import lru_cache
from typing import Any, Literal
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator



class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False
    )

    # Server Settings
    APP_NAME: str = "Clinical Audio Assessment Pipeline"
    APP_VERSION: str = "1.0.0"
    ENVIRONMENT: str = "development"
    DEBUG: bool = False
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # MongoDB Settings
    MONGODB_URI: str = "mongodb://localhost:27017"
    MONGODB_DATABASE: str = "clinical_assessments"
    MONGODB_COLLECTION: str = "assessments"
    MONGODB_TIMEOUT_MS: int = 5000

    # LLM Settings (OpenAI, Grok / xAI, Groq, Ollama, etc.)
    OPENAI_API_KEY: str = Field(default="", description="OpenAI or compatible provider API key")
    XAI_API_KEY: str = Field(default="", description="xAI Grok API key")
    GROK_API_KEY: str = Field(default="", description="Alias for Grok / Groq API key")
    GROQ_API_KEY: str = Field(default="", description="Groq Cloud API key")
    LLM_BASE_URL: str = Field(default="", description="Base URL for LLM API (e.g. https://api.x.ai/v1 or https://api.groq.com/openai/v1)")
    LLM_MODEL: str = "gpt-4o-mini"
    LLM_TEMPERATURE: float = 0.0

    # Whisper Settings
    # Modes: "openai" (OpenAI/Groq Audio API), "local" (local whisper package), "mock" (deterministic test mock)
    WHISPER_MODE: Literal["openai", "local", "mock"] = "openai"
    WHISPER_BASE_URL: str = Field(default="", description="Base URL for Whisper API")
    WHISPER_MODEL: str = "whisper-1"
    LOCAL_WHISPER_MODEL: str = "base"

    @property
    def effective_llm_api_key(self) -> str:
        """Returns active LLM API key across OpenAI, Groq, and xAI."""
        return self.OPENAI_API_KEY or self.GROQ_API_KEY or self.XAI_API_KEY or self.GROK_API_KEY

    @property
    def effective_llm_base_url(self) -> str | None:
        """Returns base URL if set or defaults based on provider key."""
        if self.LLM_BASE_URL:
            return self.LLM_BASE_URL
        if (self.GROQ_API_KEY or self.effective_llm_api_key.startswith("gsk_")):
            return "https://api.groq.com/openai/v1"
        if (self.XAI_API_KEY or self.GROK_API_KEY) and not self.OPENAI_API_KEY:
            return "https://api.x.ai/v1"
        return None

    @property
    def effective_whisper_api_key(self) -> str:
        """Returns active Whisper API key."""
        return self.OPENAI_API_KEY or self.GROQ_API_KEY or self.effective_llm_api_key

    @property
    def effective_whisper_base_url(self) -> str | None:
        """Returns active Whisper base URL."""
        if self.WHISPER_BASE_URL:
            return self.WHISPER_BASE_URL
        if self.effective_whisper_api_key.startswith("gsk_"):
            return "https://api.groq.com/openai/v1"
        return self.effective_llm_base_url


    # Audio & Extraction Validation
    CONFIDENCE_THRESHOLD: float = 0.70
    MAX_AUDIO_SIZE_BYTES: int = 50 * 1024 * 1024  # 50 MB
    ALLOWED_AUDIO_EXTENSIONS: str = ".wav"
    TEMP_UPLOAD_DIR: str = "temp_uploads"

    @property
    def allowed_extensions_list(self) -> list[str]:
        """Returns list of allowed audio file extensions."""
        exts = self.ALLOWED_AUDIO_EXTENSIONS
        if exts.startswith("[") and exts.endswith("]"):
            try:
                return json.loads(exts)
            except Exception:
                pass
        return [e.strip().lower() for e in exts.split(",") if e.strip()]




@lru_cache()
def get_settings() -> Settings:
    """Returns a cached instance of application settings."""
    return Settings()
