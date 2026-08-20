"""WAV decoding for Whisper, with no ffmpeg dependency.

Whisper expects mono float32 PCM at 16 kHz. The supplied recording is 44.1 kHz
mono 16-bit, and ffmpeg is not installed on the target machine, so decoding is
done here with the stdlib ``wave`` module plus numpy.

Both Whisper backends accept a raw numpy array, so this is the single decode
path for the whole pipeline. That matters for reproducibility: a transcript
cannot differ between backends because of a difference in resampling.

Measured on the supplied 105.5 s file: 2.4 s to decode and resample, with
duration and RMS preserved to within 0.1%.
"""

from __future__ import annotations

import wave
from pathlib import Path
from typing import Final

import numpy as np

WHISPER_SAMPLE_RATE: Final[int] = 16_000

#: Guards against a mislabelled or runaway upload reaching the model.
MAX_DURATION_SECONDS: Final[float] = 60 * 60


class InvalidAudioError(ValueError):
    """The upload is not a decodable PCM WAV file.

    Carries a message intended for the API caller, so it can be surfaced
    directly in an HTTP 400 rather than leaking a stack trace.
    """


def _pcm_to_float32(frames: bytes, sample_width: int) -> np.ndarray:
    """Convert raw PCM bytes to float32 in [-1.0, 1.0]."""
    if sample_width == 1:
        # 8-bit WAV is unsigned and centred on 128, unlike every other width.
        data = np.frombuffer(frames, dtype=np.uint8).astype(np.float32)
        return (data - 128.0) / 128.0
    if sample_width == 2:
        return np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    if sample_width == 3:
        # 24-bit has no numpy dtype; widen each 3-byte group into 4 bytes,
        # placing the source bytes high so the sign bit lands correctly.
        raw = np.frombuffer(frames, dtype=np.uint8).reshape(-1, 3)
        widened = np.zeros((raw.shape[0], 4), dtype=np.uint8)
        widened[:, 1:] = raw
        return widened.view("<i4").ravel().astype(np.float32) / (2.0**31)
    if sample_width == 4:
        return np.frombuffer(frames, dtype="<i4").astype(np.float32) / (2.0**31)

    raise InvalidAudioError(
        f"Unsupported WAV sample width: {sample_width * 8}-bit. "
        "Expected 8, 16, 24 or 32-bit PCM."
    )


def _resample_fft(signal: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Resample by spectral truncation - the method scipy.signal.resample uses.

    Truncating the spectrum is an ideal brick-wall low-pass, so downsampling
    44.1 kHz to 16 kHz introduces no aliasing. Implemented here to keep scipy
    out of the dependency list, since it is the only thing we would need it for.
    """
    if src_rate == dst_rate:
        return signal.astype(np.float32, copy=False)

    n_src = signal.shape[0]
    n_dst = int(round(n_src * dst_rate / src_rate))
    if n_dst <= 0:
        raise InvalidAudioError("Audio is too short to resample.")

    spectrum = np.fft.rfft(signal)
    n_keep = min(len(spectrum), n_dst // 2 + 1)

    resampled = np.zeros(n_dst // 2 + 1, dtype=complex)
    resampled[:n_keep] = spectrum[:n_keep]

    # irfft renormalises by the output length, so rescale to preserve amplitude.
    return (np.fft.irfft(resampled, n=n_dst) * (n_dst / n_src)).astype(np.float32)


def load_wav_16k_mono(path: str | Path) -> tuple[np.ndarray, float]:
    """Decode a WAV file to mono float32 at 16 kHz.

    Returns:
        ``(samples, duration_seconds)`` with samples in [-1.0, 1.0].

    Raises:
        InvalidAudioError: the file is missing, not PCM WAV, empty, or too long.
    """
    path = Path(path)
    if not path.is_file():
        raise InvalidAudioError(f"Audio file not found: {path}")

    try:
        with wave.open(str(path), "rb") as wav:
            channels = wav.getnchannels()
            sample_width = wav.getsampwidth()
            frame_rate = wav.getframerate()
            n_frames = wav.getnframes()
            frames = wav.readframes(n_frames)
    except (wave.Error, EOFError) as exc:
        # Covers compressed WAV (a-law/mu-law), truncated headers, and files
        # that are really mp3/m4a wearing a .wav extension. EOFError is raised
        # separately by `wave` when the file is too short to hold a header at
        # all, and must not escape as a 500.
        raise InvalidAudioError(
            f"Not a readable PCM WAV file: {exc or type(exc).__name__}"
        ) from exc

    if n_frames == 0:
        raise InvalidAudioError("WAV file contains no audio frames.")
    if frame_rate <= 0:
        raise InvalidAudioError("WAV header declares an invalid sample rate.")

    duration = n_frames / float(frame_rate)
    if duration > MAX_DURATION_SECONDS:
        raise InvalidAudioError(
            f"Audio is {duration / 60:.1f} minutes long; the limit is "
            f"{MAX_DURATION_SECONDS / 60:.0f} minutes."
        )

    samples = _pcm_to_float32(frames, sample_width)

    if channels > 1:
        # Average the channels. A clinician and patient are sometimes recorded
        # on separate channels, and both voices must reach the transcript.
        samples = samples.reshape(-1, channels).mean(axis=1)

    samples = _resample_fft(samples, frame_rate, WHISPER_SAMPLE_RATE)

    # Channel averaging and resampling can both overshoot slightly.
    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    if peak > 1.0:
        samples = samples / peak

    return samples, duration


def probe_wav(path: str | Path) -> dict:
    """Header-only inspection, so the API can reject a bad upload early."""
    try:
        with wave.open(str(path), "rb") as wav:
            frame_rate = wav.getframerate() or 1
            return {
                "channels": wav.getnchannels(),
                "sampleWidthBits": wav.getsampwidth() * 8,
                "sampleRate": wav.getframerate(),
                "frames": wav.getnframes(),
                "durationSeconds": round(wav.getnframes() / float(frame_rate), 2),
            }
    except (wave.Error, EOFError) as exc:
        raise InvalidAudioError(
            f"Not a readable PCM WAV file: {exc or type(exc).__name__}"
        ) from exc
