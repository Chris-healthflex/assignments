import io
import wave
from pathlib import Path
from app.core.config import settings
from app.core.exceptions import AudioValidationError

def validate_wav_bytes(data: bytes, filename: str) -> None:
    """Validate uploaded file is a WAV and within size limits."""
    if not filename.lower().endswith(".wav"):
        raise AudioValidationError("File must be a .wav file")
    if len(data) > settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024:
        raise AudioValidationError(f"File size exceeds {settings.MAX_UPLOAD_SIZE_MB}MB limit")
    try:
        with io.BytesIO(data) as f:
            wave.open(f, 'rb')
    except Exception:
        raise AudioValidationError("Invalid WAV file")

def get_wav_duration_seconds(file_path: str) -> float:
    """Return duration of WAV file using wave module (simple)."""
    with wave.open(file_path, 'rb') as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
        duration = frames / float(rate)
    return duration