from __future__ import annotations

import os
import threading
import wave
from pathlib import Path

import whisper

from app.config import get_settings


_model = None
_model_lock = threading.Lock()


class TranscriptionError(Exception):
    """Raised when audio transcription fails."""


def validate_wav_file(file_path: str | Path) -> None:
    """
    Validate that the supplied file is a readable WAV file.
    """

    path = Path(file_path)

    if not path.exists():
        raise TranscriptionError(f"Audio file does not exist: {path}")

    if not path.is_file():
        raise TranscriptionError(f"Audio path is not a file: {path}")

    if path.stat().st_size == 0:
        raise TranscriptionError("Audio file is empty.")

    try:
        with wave.open(str(path), "rb") as wav_file:
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            frame_rate = wav_file.getframerate()
            frames = wav_file.getnframes()

            if channels <= 0:
                raise TranscriptionError(
                    "WAV file contains no audio channels."
                )

            if sample_width <= 0:
                raise TranscriptionError(
                    "WAV file has an invalid sample width."
                )

            if frame_rate <= 0:
                raise TranscriptionError(
                    "WAV file has an invalid sample rate."
                )

            if frames <= 0:
                raise TranscriptionError(
                    "WAV file contains no audio frames."
                )

    except (wave.Error, EOFError) as exc:
        raise TranscriptionError(
            f"Invalid or unreadable WAV file: {exc}"
        ) from exc


def _get_model():
    """
    Load Whisper once and reuse it for subsequent requests.
    """

    global _model

    if _model is None:
        with _model_lock:
            if _model is None:
                settings = get_settings()

                _model = whisper.load_model(
                    settings.whisper_model
                )

    return _model


def transcribe_audio(file_path: str | Path) -> str:
    """
    Transcribe a WAV file using local Whisper.
    """

    validate_wav_file(file_path)

    settings = get_settings()

    model = _get_model()

    try:
        result = model.transcribe(
            str(file_path),
            language=settings.whisper_language,
            fp16=False,
            verbose=False,
            condition_on_previous_text=True,
        )
    except Exception as exc:
        raise TranscriptionError(
            f"Whisper transcription failed: {exc}"
        ) from exc

    text = (result.get("text") or "").strip()

    if not text:
        raise TranscriptionError(
            "Whisper returned an empty transcript."
        )

    return text
