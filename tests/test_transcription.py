import pytest
from app.services.transcription import MockWhisperTranscriber, OpenAIWhisperTranscriber


@pytest.mark.asyncio
async def test_mock_transcriber_returns_text():
    """Verify MockWhisperTranscriber returns expected transcript."""
    expected = "Patient has knee pain for three days."
    transcriber = MockWhisperTranscriber(mock_transcript=expected)
    result = await transcriber.transcribe(b"dummy_bytes", "dummy_file.wav")
    assert result == expected


def test_whisper_transcriber_initialization():
    """Verify OpenAIWhisperTranscriber initializes with configured settings."""
    transcriber = OpenAIWhisperTranscriber()
    assert transcriber.settings.WHISPER_MODEL is not None
    assert transcriber.settings.effective_whisper_base_url is not None
