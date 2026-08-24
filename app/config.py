"""Centralised configuration, all overridable via environment / .env."""
from __future__ import annotations

import os
from dataclasses import dataclass, field


def _get_bool(key: str, default: bool) -> bool:
    val = os.getenv(key)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Settings:
    # --- Whisper ---
    whisper_model: str = field(default_factory=lambda: os.getenv("WHISPER_MODEL", "small"))
    whisper_backend: str = field(
        default_factory=lambda: os.getenv("WHISPER_BACKEND", "faster-whisper")
    )
    whisper_device: str = field(default_factory=lambda: os.getenv("WHISPER_DEVICE", "cpu"))
    whisper_compute_type: str = field(
        default_factory=lambda: os.getenv("WHISPER_COMPUTE_TYPE", "int8")
    )

    # --- LLM (Ollama) ---
    llm_backend: str = field(default_factory=lambda: os.getenv("LLM_BACKEND", "ollama"))
    ollama_host: str = field(
        default_factory=lambda: os.getenv("OLLAMA_HOST", "http://localhost:11434")
    )
    ollama_model: str = field(default_factory=lambda: os.getenv("OLLAMA_MODEL", "llama3.1"))
    # When true the pipeline uses the deterministic rule-based extractor instead of
    # calling Ollama. Lets tests + the demo script run with zero external services.
    use_stub_llm: bool = field(default_factory=lambda: _get_bool("USE_STUB_LLM", False))

    # --- MongoDB ---
    mongo_uri: str = field(
        default_factory=lambda: os.getenv("MONGO_URI", "mongodb://localhost:27017")
    )
    mongo_db: str = field(default_factory=lambda: os.getenv("MONGO_DB", "clinical"))
    mongo_collection: str = field(
        default_factory=lambda: os.getenv("MONGO_COLLECTION", "assessments")
    )
    # Fall back to an in-memory store when Mongo is unreachable (tests / demo).
    allow_memory_db: bool = field(default_factory=lambda: _get_bool("ALLOW_MEMORY_DB", True))

    # --- Confidence ---
    confidence_threshold: float = field(
        default_factory=lambda: float(os.getenv("CONFIDENCE_THRESHOLD", "0.55"))
    )


settings = Settings()
