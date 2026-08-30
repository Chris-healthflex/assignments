from __future__ import annotations

import logging
import wave
from pathlib import Path

from app.config import Settings
from app.errors import BadRequestError, TranscriptionError

logger = logging.getLogger(__name__)


def validate_wav(path: Path) -> None:
    if not path.exists() or path.stat().st_size == 0:
        raise BadRequestError("Audio file is empty")

    try:
        with wave.open(str(path), "rb") as audio:
            if audio.getnframes() == 0:
                raise BadRequestError("Audio file is empty")
    except wave.Error as exc:
        raise BadRequestError("Audio file must be a valid WAV recording") from exc


def transcribe_wav(path: Path, settings: Settings) -> str:
    validate_wav(path)

    try:
        import whisper
    except ImportError as exc:
        raise TranscriptionError("Whisper is not installed") from exc

    try:
        model = whisper.load_model(settings.whisper_model)
        result = model.transcribe(str(path))
    except Exception as exc:
        logger.exception("Whisper transcription failed")
        raise TranscriptionError("Could not transcribe audio") from exc

    text = result.get("text", "") if isinstance(result, dict) else ""
    return text.strip()
