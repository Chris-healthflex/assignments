"""Audio validation."""
import pytest
from app.transcription.audio_io import validate_wav, AudioError


def test_valid_wav(wav_path):
    info = validate_wav(wav_path)
    assert info.duration_seconds > 0
    assert info.sample_rate > 0


def test_missing_file():
    with pytest.raises(AudioError):
        validate_wav("data/does_not_exist.wav")


def test_non_wav(tmp_path):
    p = tmp_path / "fake.wav"
    p.write_bytes(b"not a wav at all")
    with pytest.raises(AudioError):
        validate_wav(str(p))
