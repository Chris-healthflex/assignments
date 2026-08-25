"""Whisper transcription. Isolated from the API layer so it can be mocked in tests."""
from __future__ import annotations

import logging
import wave
from pathlib import Path
from typing import Optional, Protocol

from app.config import get_settings

logger = logging.getLogger(__name__)


class TranscriptionError(Exception):
    """Raised when audio cannot be read or transcribed."""


class Transcriber(Protocol):
    """Interface the pipeline depends on - lets tests inject a fake."""

    def transcribe(self, audio_path: str | Path) -> str: ...


def validate_wav(audio_path: str | Path) -> None:
    """Confirm the file exists and is a readable RIFF/WAVE container."""
    path = Path(audio_path)
    if not path.exists():
        raise TranscriptionError(f"Audio file not found: {path}")
    if path.suffix.lower() != ".wav":
        raise TranscriptionError(f"Expected a .wav file, got '{path.suffix}'")
    if path.stat().st_size == 0:
        raise TranscriptionError("Audio file is empty")
    try:
        with wave.open(str(path), "rb") as handle:
            if handle.getnframes() == 0:
                raise TranscriptionError("Audio file contains no frames")
    except wave.Error as exc:
        raise TranscriptionError(f"Not a valid WAV file: {exc}") from exc


class WhisperTranscriber:
    """faster-whisper backed transcriber.

    The model is loaded lazily on first use: importing this module (which the
    test suite and the FastAPI app both do at startup) must not pull a
    multi-hundred-megabyte model into memory.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        compute_type: Optional[str] = None,
    ) -> None:
        settings = get_settings()
        self.model_name = model_name or settings.whisper_model
        self.device = device or settings.whisper_device
        self.compute_type = compute_type or settings.whisper_compute_type
        self._model = None

    def _load_model(self):
        if self._model is None:
            try:
                from faster_whisper import WhisperModel
            except ImportError as exc:  # pragma: no cover
                raise TranscriptionError(
                    "faster-whisper is not installed. Run: pip install faster-whisper"
                ) from exc
            logger.info("Loading Whisper model '%s' on %s", self.model_name, self.device)
            self._model = WhisperModel(
                self.model_name, device=self.device, compute_type=self.compute_type
            )
        return self._model

    def transcribe(self, audio_path: str | Path) -> str:
        validate_wav(audio_path)
        model = self._load_model()
        try:
            segments, _info = model.transcribe(str(audio_path), beam_size=5, vad_filter=True)
            text = " ".join(segment.text.strip() for segment in segments).strip()
        except TranscriptionError:
            raise
        except Exception as exc:
            raise TranscriptionError(f"Transcription failed: {exc}") from exc

        if not text:
            raise TranscriptionError("Transcription produced no text")
        logger.info("Transcribed %s characters", len(text))
        return text
