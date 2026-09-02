"""Unit tests for Whisper audio transcription service."""

from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
from openai import OpenAIError

from app.config import settings
from app.services.transcriber import (
    AudioValidationError,
    TranscriptionError,
    WhisperTranscriber,
    transcribe_audio,
)


@pytest.fixture
def temp_wav_file(tmp_path: Path) -> Path:
    """Create a temporary dummy WAV file."""
    wav_file = tmp_path / "test_session.wav"
    # Write dummy non-empty binary content
    wav_file.write_bytes(b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x44\xac\x00\x00data\x00\x00\x00\x00")
    return wav_file


@pytest.fixture
def mock_openai_client() -> MagicMock:
    """Create a mocked OpenAI client with successful transcription response."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "Patient presents with left knee pain and restricted range of motion."
    mock_client.audio.transcriptions.create.return_value = mock_response
    return mock_client


def test_transcription_successful(temp_wav_file: Path, mock_openai_client: MagicMock):
    """Test 1: Successful transcription returns expected transcript string."""
    transcriber = WhisperTranscriber(client=mock_openai_client)
    result = transcriber.transcribe(temp_wav_file)

    assert isinstance(result, str)
    assert result == "Patient presents with left knee pain and restricted range of motion."
    mock_openai_client.audio.transcriptions.create.assert_called_once()


def test_empty_audio_file_rejected(tmp_path: Path, mock_openai_client: MagicMock):
    """Test 2: Empty (0 bytes) audio file is rejected with AudioValidationError."""
    empty_file = tmp_path / "empty.wav"
    empty_file.write_bytes(b"")

    transcriber = WhisperTranscriber(client=mock_openai_client)
    with pytest.raises(AudioValidationError) as exc_info:
        transcriber.transcribe(empty_file)

    assert "empty" in str(exc_info.value).lower()
    mock_openai_client.audio.transcriptions.create.assert_not_called()


def test_missing_audio_file_handled_clearly(tmp_path: Path, mock_openai_client: MagicMock):
    """Test 3: Missing audio file path raises AudioValidationError."""
    non_existent = tmp_path / "does_not_exist.wav"

    transcriber = WhisperTranscriber(client=mock_openai_client)
    with pytest.raises(AudioValidationError) as exc_info:
        transcriber.transcribe(non_existent)

    assert "not found" in str(exc_info.value).lower()
    mock_openai_client.audio.transcriptions.create.assert_not_called()


def test_unsupported_file_extension_rejected(tmp_path: Path, mock_openai_client: MagicMock):
    """Test 4: Non-audio file extension is rejected with AudioValidationError."""
    text_file = tmp_path / "notes.txt"
    text_file.write_text("dummy notes")

    transcriber = WhisperTranscriber(client=mock_openai_client)
    with pytest.raises(AudioValidationError) as exc_info:
        transcriber.transcribe(text_file)

    assert "unsupported audio format" in str(exc_info.value).lower()
    mock_openai_client.audio.transcriptions.create.assert_not_called()


def test_whisper_api_error_wrapped(temp_wav_file: Path, mock_openai_client: MagicMock):
    """Test 5: OpenAI API errors are converted into TranscriptionError."""
    mock_openai_client.audio.transcriptions.create.side_effect = OpenAIError("Rate limit exceeded")

    transcriber = WhisperTranscriber(client=mock_openai_client)
    with pytest.raises(TranscriptionError) as exc_info:
        transcriber.transcribe(temp_wav_file)

    assert "Whisper API transcription error" in str(exc_info.value)
    assert "Rate limit exceeded" in str(exc_info.value)


def test_configured_whisper_model_used(temp_wav_file: Path, mock_openai_client: MagicMock):
    """Test 6: Configured Whisper model is passed to the API call."""
    custom_model = "whisper-custom-model"
    transcriber = WhisperTranscriber(model=custom_model, client=mock_openai_client)
    transcriber.transcribe(temp_wav_file)

    call_kwargs = mock_openai_client.audio.transcriptions.create.call_args.kwargs
    assert call_kwargs["model"] == custom_model


def test_api_key_obtained_from_config(temp_wav_file: Path):
    """Test 7: API key is obtained from configuration and unconfigured key raises error."""
    # When api_key is empty or placeholder, accessing client raises TranscriptionError
    transcriber = WhisperTranscriber(api_key="")
    with pytest.raises(TranscriptionError) as exc_info:
        _ = transcriber.client

    assert "OpenAI API key is not configured" in str(exc_info.value)


def test_audio_file_resources_properly_closed(temp_wav_file: Path, mock_openai_client: MagicMock):
    """Test 8: Audio file handle is properly closed after transcription."""
    file_handle_mock = None

    original_open = open

    def spy_open(*args, **kwargs):
        nonlocal file_handle_mock
        file_obj = original_open(*args, **kwargs)
        file_handle_mock = file_obj
        return file_obj

    with patch("builtins.open", side_effect=spy_open):
        transcriber = WhisperTranscriber(client=mock_openai_client)
        transcriber.transcribe(temp_wav_file)

    assert file_handle_mock is not None
    assert file_handle_mock.closed


def test_convenience_transcribe_audio_function(temp_wav_file: Path, mock_openai_client: MagicMock):
    """Test 9: transcribe_audio convenience function works as expected."""
    transcriber = WhisperTranscriber(client=mock_openai_client)
    result = transcribe_audio(temp_wav_file, transcriber=transcriber)
    assert result == "Patient presents with left knee pain and restricted range of motion."


def test_real_clinical_assessment_wav_if_available():
    """Test 10 (Optional / Integration): Transcribes real clinical_assessment.wav if key and file exist."""
    audio_path = Path("clinical_assessment.wav")
    api_key = settings.OPENAI_API_KEY.strip()

    if not audio_path.exists():
        pytest.skip("clinical_assessment.wav not present in workspace root (skipping live test)")

    if not api_key or api_key in {"your_openai_api_key_here", "mock_key"}:
        pytest.skip("OPENAI_API_KEY not configured for live Whisper API call (skipping live test)")

    try:
        transcriber = WhisperTranscriber()
        transcript = transcriber.transcribe(audio_path)
    except TranscriptionError as exc:
        if "insufficient_quota" in str(exc) or "credit_balance_exhausted" in str(exc) or "429" in str(exc):
            pytest.skip(f"OpenAI API quota exhausted (skipping live test): {str(exc)}")
        raise

    assert isinstance(transcript, str)
    assert len(transcript) > 50
    print(f"\n[Real Transcription Result Preview]:\n{transcript[:300]}...")
