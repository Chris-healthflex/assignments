"""Tests for the ffmpeg-free audio front end.

Fixture WAVs are synthesised here with the standard library rather than checked
in as binaries, and nothing in this module imports whisper or torch: the array
functions under test are deliberately independent of the model.
"""

import wave
from pathlib import Path

import numpy as np
import pytest

from clinical_assessment.errors import AudioDecodeError
from clinical_assessment.transcription import (
    decode_wav,
    resample,
    to_float32,
)


def write_wav(
    path: Path,
    samples: np.ndarray,
    sample_rate: int = 44_100,
    channel_count: int = 1,
    sample_width: int = 2,
) -> Path:
    """Write interleaved samples to a PCM WAV file and return its path."""
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(channel_count)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(samples.astype("<i2").tobytes())
    return path


def test_decode_wav_returns_mono_samples_and_rate(tmp_path: Path) -> None:
    written = np.array([0, 1000, -1000, 32767, -32768], dtype=np.int16)
    wav_path = write_wav(tmp_path / "mono.wav", written, sample_rate=16_000)

    samples, sample_rate = decode_wav(wav_path)

    assert (samples.tolist(), sample_rate) == (written.tolist(), 16_000)


def test_resample_44100_to_16000_scales_length_proportionally() -> None:
    samples = np.linspace(-1.0, 1.0, 44_100, dtype=np.float32)

    resampled = resample(samples, 44_100, 16_000)

    assert resampled.size == 16_000


def test_resample_preserves_the_signal_endpoints() -> None:
    samples = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float32)

    resampled = resample(samples, 8_000, 16_000)

    assert (resampled[0], resampled[-1]) == (0.0, 1.0)


def test_stereo_input_is_downmixed_to_mono(tmp_path: Path) -> None:
    # Interleaved L/R pairs; each pair averages to 100.
    interleaved = np.array([0, 200, 50, 150, -100, 300], dtype=np.int16)
    wav_path = write_wav(tmp_path / "stereo.wav", interleaved, channel_count=2)

    samples, _ = decode_wav(wav_path)

    assert samples.tolist() == [100, 100, 100]


def test_to_float32_maps_int16_extremes_into_unit_range() -> None:
    extremes = np.array([-32768, 0, 32767], dtype=np.int16)

    converted = to_float32(extremes)

    assert converted.tolist() == pytest.approx([-1.0, 0.0, 0.999969], abs=1e-5)


def test_decode_wav_raises_on_non_wav_bytes(tmp_path: Path) -> None:
    not_audio = tmp_path / "notes.txt"
    not_audio.write_bytes(b"this is plainly not a RIFF header")

    with pytest.raises(AudioDecodeError, match="not readable WAV audio"):
        decode_wav(not_audio)


def test_decode_wav_raises_on_zero_length_audio(tmp_path: Path) -> None:
    wav_path = write_wav(tmp_path / "silent.wav", np.array([], dtype=np.int16))

    with pytest.raises(AudioDecodeError, match="no audio frames"):
        decode_wav(wav_path)


def test_decode_wav_raises_on_unsupported_sample_width(tmp_path: Path) -> None:
    wav_path = tmp_path / "eight_bit.wav"
    with wave.open(str(wav_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(1)
        wav_file.setframerate(16_000)
        wav_file.writeframes(b"\x00\x40\x80")

    with pytest.raises(AudioDecodeError, match="only 16-bit PCM WAV"):
        decode_wav(wav_path)


@pytest.mark.parametrize(
    ("source_rate", "target_rate"), [(0, 16_000), (44_100, 0), (-1, 16_000)]
)
def test_resample_raises_on_non_positive_rate(
    source_rate: int, target_rate: int
) -> None:
    samples = np.array([0.0, 1.0], dtype=np.float32)

    with pytest.raises(ValueError, match="must be positive"):
        resample(samples, source_rate, target_rate)
