import pytest
from app.services.audio_validator import AudioValidator
from app.core.errors import AudioValidationError
from tests.conftest import create_mock_wav_bytes


def test_valid_wav_accepted():
    """Verify that a standard mono 16kHz WAV byte stream is accepted."""
    validator = AudioValidator()
    wav_bytes = create_mock_wav_bytes(duration_sec=2.0, sample_rate=16000)
    # Should not raise
    validator.validate_wav_bytes(wav_bytes, "valid_test.wav")


def test_invalid_extension_rejected():
    """Verify that non-WAV extensions are rejected with AudioValidationError."""
    validator = AudioValidator()
    wav_bytes = create_mock_wav_bytes(duration_sec=1.0)
    with pytest.raises(AudioValidationError) as exc:
        validator.validate_wav_bytes(wav_bytes, "test_document.pdf")
    assert "Invalid file extension" in str(exc.value)


def test_empty_wav_rejected():
    """Verify that empty file uploads are rejected."""
    validator = AudioValidator()
    with pytest.raises(AudioValidationError) as exc:
        validator.validate_wav_bytes(b"", "empty.wav")
    assert "empty" in str(exc.value).lower()


def test_corrupted_header_rejected():
    """Verify that random bytes disguised as .wav are rejected by RIFF parser."""
    validator = AudioValidator()
    corrupt_bytes = b"NOT_A_RIFF_HEADER_12345678901234567890"
    with pytest.raises(AudioValidationError) as exc:
        validator.validate_wav_bytes(corrupt_bytes, "corrupt.wav")
    assert "too small" in str(exc.value).lower() or "riff" in str(exc.value).lower() or "corrupt" in str(exc.value).lower()


def test_file_size_limit_rejected():
    """Verify that files exceeding MAX_AUDIO_SIZE_BYTES are rejected."""
    validator = AudioValidator()
    large_bytes = b"\x00" * (53 * 1024 * 1024)  # 53 MB
    with pytest.raises(AudioValidationError) as exc:
        validator.validate_wav_bytes(large_bytes, "huge.wav")
    assert "exceeds maximum allowed size" in str(exc.value)
