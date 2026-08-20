from __future__ import annotations

import logging
from functools import lru_cache
from typing import Literal

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

_LOCAL_WHISPER_SIZES = {
    "tiny", "tiny.en", "base", "base.en", "small", "small.en",
    "medium", "medium.en", "large", "large-v1", "large-v2", "large-v3",
    "large-v3-turbo", "distil-large-v3",
}
GROQ_DEFAULT_WHISPER = "whisper-large-v3-turbo"

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )


    whisper_backend: Literal["groq", "openai", "local"] = "groq"
    whisper_model: str = GROQ_DEFAULT_WHISPER
    whisper_language: str | None = None  


    whisper_downsample: bool = True
    whisper_max_request_bytes: int = 25 * 1024 * 1024

    whisper_device: str = "cpu"
    whisper_compute_type: str = "int8"

    llm_provider: Literal["groq", "anthropic", "openai", "stub"] = "groq"
    llm_model: str = "openai/gpt-oss-120b"
    llm_max_tokens: int = 8000

    llm_structured_method: Literal["function_calling", "json_schema", "json_mode"] = (
        "function_calling"
    )

    groq_api_key: str | None = None
    anthropic_api_key: str | None = None
    openai_api_key: str | None = None

    confidence_threshold: float = 0.55
    max_extraction_attempts: int = 2

    mongodb_uri: str = "mongodb://localhost:27017"
    mongodb_db: str = "stance_health"
    mongodb_collection: str = "stance"

    mongodb_timeout_ms: int = 8000

    max_upload_bytes: int = 100 * 1024 * 1024  # 100 MB

    @model_validator(mode="after")
    def _fix_stale_local_model_name(self) -> "Settings":
        if self.whisper_backend == "groq" and self.whisper_model in _LOCAL_WHISPER_SIZES:
            logger.warning(
                "WHISPER_MODEL=%r is a local faster-whisper size, not a Groq "
                "model id; using %r instead.",
                self.whisper_model, GROQ_DEFAULT_WHISPER,
            )
            object.__setattr__(self, "whisper_model", GROQ_DEFAULT_WHISPER)
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
