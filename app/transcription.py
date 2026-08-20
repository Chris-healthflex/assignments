"""Whisper transcription.

Scaffold: the local faster-whisper backend is wired end-to-end; the hosted
"api" backend is a placeholder until we decide whether audio leaves the box
(PHI -- likely local-only in prod).
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

from app.config import get_settings
from app.schemas import TranscriptionResult

logger = logging.getLogger(__name__)


class TranscriptionError(RuntimeError):
    """Raised when audio cannot be transcribed."""


@lru_cache
def _load_model():
    """Load the Whisper model once per process (it is several hundred MB)."""
    from faster_whisper import WhisperModel  # imported lazily: heavy dependency

    settings = get_settings()
    logger.info(
        "Loading Whisper model=%s device=%s",
        settings.whisper_model,
        settings.whisper_device,
    )
    return WhisperModel(
        settings.whisper_model,
        device=settings.whisper_device,
        compute_type="int8" if settings.whisper_device == "cpu" else "float16",
    )


def transcribe(audio_path: str | Path) -> TranscriptionResult:
    """Transcribe an audio file into text plus timed segments."""
    path = Path(audio_path)
    if not path.is_file():
        raise TranscriptionError(f"Audio file not found: {path}")

    settings = get_settings()
    if settings.whisper_backend == "api":
        raise NotImplementedError("Hosted Whisper backend not wired yet (see D2).")

    model = _load_model()
    segments, info = model.transcribe(
        str(path),
        language=settings.whisper_language,
        vad_filter=True,
    )

    collected = [
        {"start": s.start, "end": s.end, "text": s.text.strip()} for s in segments
    ]
    text = " ".join(s["text"] for s in collected).strip()
    if not text:
        raise TranscriptionError(f"Whisper produced no text for {path.name}")

    return TranscriptionResult(
        text=text,
        language=getattr(info, "language", None),
        duration_sec=getattr(info, "duration", None),
        segments=collected,
    )
