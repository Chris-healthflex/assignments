import io
import math
from pathlib import Path
from typing import Union, BinaryIO
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


class AudioValidationError(ValueError):
    """Raised when an uploaded audio file is invalid or not a supported WAV format."""
    pass


def validate_wav_header(stream_or_bytes: Union[bytes, BinaryIO]) -> None:
    """Verify RIFF and WAVE magic bytes without fully decoding."""
    if isinstance(stream_or_bytes, bytes):
        header = stream_or_bytes[:12]
    else:
        current_pos = stream_or_bytes.tell()
        header = stream_or_bytes.read(12)
        stream_or_bytes.seek(current_pos)

    if len(header) < 12:
        raise AudioValidationError("File is too small to be a valid WAV file.")
    if not (header[:4] == b"RIFF" and header[8:12] == b"WAVE"):
        raise AudioValidationError("Invalid audio format: file must be a standard RIFF/WAVE audio file.")


def load_and_resample_wav(
    audio_source: Union[str, Path, bytes, BinaryIO],
    target_sr: int = 16000,
) -> np.ndarray:
    """
    Decodes a WAV file and resamples it to target_sr mono float32 numpy array.
    Strictly uses soundfile and scipy.signal.resample_poly, eliminating any system ffmpeg dependency.
    """
    if isinstance(audio_source, bytes):
        validate_wav_header(audio_source)
        source = io.BytesIO(audio_source)
    elif hasattr(audio_source, "read"):
        validate_wav_header(audio_source)
        source = audio_source
    else:
        path = Path(audio_source)
        if not path.exists():
            raise AudioValidationError(f"Audio file not found: {path}")
        with open(path, "rb") as f:
            validate_wav_header(f)
        source = str(path)

    try:
        data, sample_rate = sf.read(source, dtype="float32")
    except Exception as exc:
        raise AudioValidationError(f"Failed to read WAV audio: {exc}") from exc

    # Convert to mono if multi-channel
    if data.ndim > 1:
        data = np.mean(data, axis=1)

    # Resample if sample rate doesn't match target_sr
    if sample_rate != target_sr:
        g = math.gcd(target_sr, sample_rate)
        up = target_sr // g
        down = sample_rate // g
        data = resample_poly(data, up, down).astype(np.float32)

    return data
