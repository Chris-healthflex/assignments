from __future__ import annotations

import os
import tempfile

from app.core.config import get_settings


class TranscriptionError(Exception):
    """Raised when the audio cannot be transcribed."""


_model = None
_loaded_model_size: str | None = None


def _get_model():
    """Lazily load and cache the local Whisper model."""
    global _model, _loaded_model_size

    settings = get_settings()
    model_size = settings.whisper_model_size

    if _model is not None and _loaded_model_size == model_size:
        return _model

    try:
        import whisper  # openai-whisper package
    except ImportError as exc:
        raise TranscriptionError(
            "The 'openai-whisper' package is not installed. "
            "Run `pip install -r requirements.txt` (it also requires "
            "ffmpeg to be installed on your system)."
        ) from exc

    _model = whisper.load_model(model_size)
    _loaded_model_size = model_size
    return _model


def transcribe_wav(file_bytes: bytes, filename: str = "audio.wav") -> str:
    """
    Transcribe a WAV file's raw bytes into text using a local Whisper model.

    Args:
        file_bytes: Raw bytes of the WAV file.
        filename: Original filename (unused beyond logging/temp-file suffix).

    Returns:
        The transcript text.

    Raises:
        TranscriptionError: if the file is empty, whisper/ffmpeg isn't
            available, or transcription fails.
    """
    if not file_bytes:
        raise TranscriptionError("Received an empty audio file.")

    model = _get_model()

    # openai-whisper (and the ffmpeg call it shells out to) needs a real
    # file path, not in-memory bytes.
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        result = model.transcribe(tmp_path)
    except Exception as exc:  # noqa: BLE001 - surface as a domain error
        raise TranscriptionError(
            f"Local Whisper transcription failed: {exc}. "
            "Make sure ffmpeg is installed and on your PATH."
        ) from exc
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass

    transcript = (result.get("text") or "").strip()
    if not transcript:
        raise TranscriptionError("Whisper returned an empty transcript.")

    return transcript
