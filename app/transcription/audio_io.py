"""Audio validation / inspection.

Kept dependency-light: uses the stdlib `wave` module to validate a WAV and read
its duration before we hand it to Whisper. Non-WAV or corrupt files fail here with
a clear error instead of deep inside the model.
"""
from __future__ import annotations

import contextlib
import os
import wave
from dataclasses import dataclass


class AudioError(ValueError):
    """Raised when an audio file is missing, unreadable, or not a valid WAV."""


@dataclass
class AudioInfo:
    path: str
    duration_seconds: float
    sample_rate: int
    channels: int
    frames: int


def validate_wav(path: str) -> AudioInfo:
    """Validate that `path` is a readable WAV and return its metadata."""
    if not os.path.exists(path):
        raise AudioError(f"Audio file not found: {path}")
    if os.path.getsize(path) == 0:
        raise AudioError(f"Audio file is empty: {path}")
    try:
        with contextlib.closing(wave.open(path, "rb")) as wf:
            rate = wf.getframerate()
            frames = wf.getnframes()
            channels = wf.getnchannels()
            if rate <= 0 or frames <= 0:
                raise AudioError(f"WAV has no audio content: {path}")
            duration = frames / float(rate)
    except wave.Error as exc:  # not a valid WAV container
        raise AudioError(f"Not a valid WAV file ({exc}): {path}") from exc
    return AudioInfo(
        path=path,
        duration_seconds=duration,
        sample_rate=rate,
        channels=channels,
        frames=frames,
    )
