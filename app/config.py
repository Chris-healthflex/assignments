"""Application settings, loaded from environment / .env."""

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # App
    app_name: str = "v2n-first-assessment"
    env: Literal["local", "dev", "prod"] = "local"
    log_level: str = "INFO"

    # MongoDB
    mongodb_uri: str = "mongodb://localhost:27017"
    mongodb_db: str = "v2n"
    mongodb_assessments_collection: str = "first_assessments"
    # Short on purpose. A wrong URI or a paused Atlas cluster should fail the
    # request in seconds; the driver's 30s default hangs the worker instead.
    mongodb_timeout_ms: int = 5000

    # Whisper
    whisper_backend: Literal["local", "api"] = "local"
    # "medium" earns its cost on clinical speech: on the sample recording
    # "small" produced "ankle dosa flexion" and a degree symbol, "medium" gives
    # "ankle dorsiflexion" and the word "degrees". ~3 min on CPU for 105s of
    # audio, but the transcript cache means that is paid once, not per run.
    whisper_model: str = "medium"
    whisper_device: str = "cpu"
    whisper_language: str | None = "en"

    # Extraction agent
    google_api_key: str | None = None
    # gemini-3.1-flash-lite: the free tier on gemini-2.5-flash and 3.7-flash is
    # 5 requests/minute and a few dozen per day, which three runs exhaust. The
    # lite tier has real headroom and handled the citation discipline fine.
    extraction_model: str = "gemini-3.1-flash-lite"
    # Zero temperature is not a style preference here: extraction must be
    # reproducible, and sampling is a source of invented detail.
    extraction_temperature: float = 0.0
    extraction_max_retries: int = 2
    # The Gemini free tier allows 5 requests/minute. The agent makes 3 calls per
    # extraction (subjective / objective / plan) plus up to 2 repair calls, so
    # without client-side pacing the fan-out 429s on arrival. Held one under the
    # limit rather than at it: the limiter paces our calls, it cannot see the
    # retries the SDK fires on its own. Raise this if you are on a paid key.
    extraction_requests_per_minute: float = 4.0
    # Below this, a field is reported in the 422 rather than returned silently.
    # 0.6 keeps confidently-heard values and catches the misheard ones; it is a
    # tuning knob, not a law, which is why it lives in config.
    extraction_confidence_threshold: float = 0.6


@lru_cache
def get_settings() -> Settings:
    """Cached accessor so settings are parsed once per process."""
    return Settings()
