"""Application settings, sourced from environment variables / .env.

Every externally-variable decision - which LLM, which Whisper backend, which
MongoDB, what confidence bar - is an environment variable, so the same code
runs offline on this laptop and against hosted services in CI.

Two defaults below are hardware constraints rather than preferences, and are
documented as such because raising them silently breaks the pipeline:

* ``llm_model`` must fit entirely in the GPU's 4 GB of VRAM. A larger model
  makes Ollama attempt a hybrid GPU/CPU offload, which crashes on this
  GTX 1650 (Turing) inside the CUDA ``ggml_cuda_kernel_can_use_pdl`` path.
* ``whisper_device`` stays on the CPU so transcription never competes with the
  LLM for that same VRAM.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

#: The recording supplied with the assignment.
SAMPLE_WAV = DATA_DIR / "clinical_assessment.wav"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---- Service ---------------------------------------------------------
    app_name: str = "Structured Clinical Assessment Form Filler"
    api_version: str = "1.0.0"

    # ---- Transcription ---------------------------------------------------
    whisper_backend: Literal["faster", "openai"] = "faster"
    whisper_model: str = "small"
    whisper_device: str = "cpu"             # see module docstring
    whisper_compute_type: str = "int8"      # ~4x faster than float32 on CPU
    whisper_language: str | None = "en"     # None => auto-detect
    whisper_beam_size: int = 5

    # ---- Extraction LLM --------------------------------------------------
    llm_provider: Literal["ollama", "anthropic", "openai"] = "ollama"
    llm_model: str = ""                     # "" => per-provider default below
    llm_temperature: float = 0.0            # deterministic: never invent values
    llm_max_retries: int = 2
    ollama_base_url: str = "http://localhost:11434"
    anthropic_api_key: str = ""
    openai_api_key: str = ""

    # ---- Confidence gate -------------------------------------------------
    confidence_threshold: float = 0.55

    # ---- Storage ---------------------------------------------------------
    mongodb_uri: str = "mongodb://localhost:27017"
    mongodb_db: str = "clinical_assessments"
    mongodb_collection: str = "first_assessments"
    mongodb_timeout_ms: int = 5_000

    # ---- Uploads ---------------------------------------------------------
    max_upload_bytes: int = 200 * 1024 * 1024

    @property
    def default_llm_model(self) -> str:
        """Resolved model name, falling back to a sane per-provider default."""
        if self.llm_model:
            return self.llm_model
        return {
            "ollama": "qwen2.5:3b-instruct",   # must fit in 4 GB VRAM
            "anthropic": "claude-sonnet-5",
            "openai": "gpt-4o",
        }[self.llm_provider]


@lru_cache
def get_settings() -> Settings:
    """Cached: Whisper weights and the Mongo client must not be rebuilt per request."""
    return Settings()
