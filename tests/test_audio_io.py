"""Tests for the ffmpeg-free WAV decode path.

Fixtures are synthesised rather than checked in, so the suite runs anywhere and
covers sample widths the supplied recording does not exercise.
"""

from __future__ import annotations

import wave

import numpy as np
import pytest

from app.transcription.audio_io import (
    InvalidAudioError,
    WHISPER_SAMPLE_RATE,
    load_wav_16k_mono,
    probe_wav,
)


def write_wav(path, samples: np.ndarray, rate: int, width: int = 2, channels: int = 1):
    """Write float samples in [-1, 1] as PCM WAV of the given width."""
    if width == 1:
        data = ((samples * 127.0) + 128.0).clip(0, 255).astype(np.uint8).tobytes()
    elif width == 2:
        data = (samples * 32767.0).clip(-32768, 32767).astype("<i2").tobytes()
    elif width == 4:
        data = (samples * (2**31 - 1)).clip(-(2**31), 2**31 - 1).astype("<i4").tobytes()
    else:
        raise ValueError(f"unsupported width {width}")

    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(width)
        wav.setframerate(rate)
        wav.writeframes(data)
    return path


def tone(freq: float, seconds: float, rate: int) -> np.ndarray:
    t = np.arange(int(seconds * rate)) / rate
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def dominant_frequency(signal: np.ndarray, rate: int) -> float:
    spectrum = np.abs(np.fft.rfft(signal))
    return float(np.fft.rfftfreq(len(signal), 1 / rate)[int(np.argmax(spectrum))])


# --------------------------------------------------------------------------
# Correctness of the resampler
# --------------------------------------------------------------------------
def test_resampling_preserves_pitch(tmp_path):
    """The strongest check: a 440 Hz tone must still be 440 Hz at 16 kHz.

    A shape-only assertion would pass even with a badly aliasing resampler.
    """
    path = write_wav(tmp_path / "tone.wav", tone(440, 2.0, 44100), 44100)
    samples, duration = load_wav_16k_mono(path)

    assert duration == pytest.approx(2.0, abs=0.01)
    assert samples.shape[0] == pytest.approx(2.0 * WHISPER_SAMPLE_RATE, rel=0.01)
    assert dominant_frequency(samples, WHISPER_SAMPLE_RATE) == pytest.approx(440, abs=2)


def test_resampling_preserves_amplitude(tmp_path):
    path = write_wav(tmp_path / "tone.wav", tone(300, 1.0, 44100), 44100)
    samples, _ = load_wav_16k_mono(path)

    rms = float(np.sqrt((samples**2).mean()))
    assert rms == pytest.approx(0.5 / np.sqrt(2), rel=0.05)   # RMS of a 0.5 sine
    assert float(np.max(np.abs(samples))) <= 1.0


def test_already_16k_audio_is_passed_through(tmp_path):
    path = write_wav(tmp_path / "tone.wav", tone(440, 1.0, 16000), 16000)
    samples, _ = load_wav_16k_mono(path)
    assert samples.shape[0] == 16000
    assert samples.dtype == np.float32


def test_output_is_always_float32_and_finite(tmp_path):
    path = write_wav(tmp_path / "tone.wav", tone(1000, 0.5, 44100), 44100)
    samples, _ = load_wav_16k_mono(path)
    assert samples.dtype == np.float32
    assert np.isfinite(samples).all()


# --------------------------------------------------------------------------
# Channels and sample widths
# --------------------------------------------------------------------------
def test_stereo_is_downmixed_to_mono(tmp_path):
    """Clinician and patient are sometimes on separate channels; keep both."""
    left = tone(440, 1.0, 44100)
    right = tone(880, 1.0, 44100)
    interleaved = np.empty(left.size * 2, dtype=np.float32)
    interleaved[0::2] = left
    interleaved[1::2] = right

    path = write_wav(tmp_path / "stereo.wav", interleaved, 44100, channels=2)
    samples, duration = load_wav_16k_mono(path)

    assert duration == pytest.approx(1.0, abs=0.01)
    assert samples.ndim == 1
    # Both source voices must survive the downmix.
    spectrum = np.abs(np.fft.rfft(samples))
    freqs = np.fft.rfftfreq(len(samples), 1 / WHISPER_SAMPLE_RATE)
    for expected in (440, 880):
        band = spectrum[(freqs > expected - 15) & (freqs < expected + 15)]
        assert band.max() > spectrum.mean() * 10


@pytest.mark.parametrize("width", [1, 2, 4])
def test_supported_sample_widths(tmp_path, width):
    path = write_wav(tmp_path / f"w{width}.wav", tone(440, 0.5, 16000), 16000, width=width)
    samples, _ = load_wav_16k_mono(path)
    assert samples.dtype == np.float32
    # 8-bit is coarse, so allow a wider tolerance on pitch.
    assert dominant_frequency(samples, WHISPER_SAMPLE_RATE) == pytest.approx(440, abs=8)


def test_eight_bit_is_centred_correctly(tmp_path):
    """8-bit WAV is unsigned around 128; a silent file must decode near zero."""
    path = write_wav(tmp_path / "quiet.wav", np.zeros(16000, dtype=np.float32), 16000, width=1)
    samples, _ = load_wav_16k_mono(path)
    assert abs(float(samples.mean())) < 0.02


# --------------------------------------------------------------------------
# Rejection paths - these become HTTP 400s
# --------------------------------------------------------------------------
def test_missing_file_is_rejected(tmp_path):
    with pytest.raises(InvalidAudioError, match="not found"):
        load_wav_16k_mono(tmp_path / "nope.wav")


def test_non_wav_file_is_rejected(tmp_path):
    """An mp3 renamed to .wav must fail cleanly, not crash."""
    fake = tmp_path / "actually_mp3.wav"
    fake.write_bytes(b"ID3\x04\x00\x00\x00\x00\x00\x00fake mp3 payload")
    with pytest.raises(InvalidAudioError, match="Not a readable PCM WAV"):
        load_wav_16k_mono(fake)


def test_empty_wav_is_rejected(tmp_path):
    path = write_wav(tmp_path / "empty.wav", np.zeros(0, dtype=np.float32), 16000)
    with pytest.raises(InvalidAudioError, match="no audio frames"):
        load_wav_16k_mono(path)


def test_truncated_file_is_rejected(tmp_path):
    """`wave` raises EOFError (not wave.Error) when the header is incomplete.

    Regression test: this used to escape as a 500 instead of a clean 400.
    """
    stub = tmp_path / "truncated.wav"
    stub.write_bytes(b"RIFF")
    with pytest.raises(InvalidAudioError, match="Not a readable PCM WAV"):
        load_wav_16k_mono(stub)


def test_error_message_is_safe_to_show_a_caller(tmp_path):
    """InvalidAudioError text is surfaced in HTTP 400 bodies."""
    fake = tmp_path / "bad.wav"
    fake.write_bytes(b"not audio at all")
    with pytest.raises(InvalidAudioError) as exc:
        load_wav_16k_mono(fake)
    assert "Traceback" not in str(exc.value)
    assert str(exc.value)


# --------------------------------------------------------------------------
# Header probe
# --------------------------------------------------------------------------
def test_probe_reports_header_without_decoding(tmp_path):
    path = write_wav(tmp_path / "probe.wav", tone(440, 3.0, 44100), 44100)
    info = probe_wav(path)
    assert info == {
        "channels": 1,
        "sampleWidthBits": 16,
        "sampleRate": 44100,
        "frames": 3 * 44100,
        "durationSeconds": 3.0,
    }


def test_probe_rejects_non_wav(tmp_path):
    fake = tmp_path / "bad.wav"
    fake.write_bytes(b"nope")
    with pytest.raises(InvalidAudioError):
        probe_wav(fake)


# --------------------------------------------------------------------------
# The supplied recording
# --------------------------------------------------------------------------
def test_supplied_recording_decodes(tmp_path):
    from app.config import SAMPLE_WAV

    if not SAMPLE_WAV.exists():
        pytest.skip("sample recording not present")

    info = probe_wav(SAMPLE_WAV)
    assert info["sampleRate"] == 44100
    assert info["channels"] == 1

    samples, duration = load_wav_16k_mono(SAMPLE_WAV)
    assert duration == pytest.approx(105.55, abs=0.1)
    assert samples.shape[0] == pytest.approx(duration * WHISPER_SAMPLE_RATE, rel=0.01)
    assert np.isfinite(samples).all()
    assert float(np.max(np.abs(samples))) > 0.1     # not silence
