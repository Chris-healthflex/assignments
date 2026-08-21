from pathlib import Path
from unittest.mock import MagicMock

import pytest
from openai import APIError

from app.services.transcription import TranscriptionError, transcribe_audio


def _fake_client(text: str) -> MagicMock:
    client = MagicMock()
    client.audio.transcriptions.create.return_value = MagicMock(text=text)
    return client


def test_transcribe_audio_returns_stripped_text(tmp_path: Path):
    wav_path = tmp_path / "session.wav"
    wav_path.write_bytes(b"RIFF....WAVEfmt ")

    client = _fake_client("  Patient reports lower back pain.  ")

    result = transcribe_audio(wav_path, client=client)

    assert result == "Patient reports lower back pain."
    client.audio.transcriptions.create.assert_called_once()


def test_transcribe_audio_raises_on_empty_transcript(tmp_path: Path):
    wav_path = tmp_path / "session.wav"
    wav_path.write_bytes(b"RIFF....WAVEfmt ")

    client = _fake_client("   ")

    with pytest.raises(TranscriptionError):
        transcribe_audio(wav_path, client=client)


def test_transcribe_audio_wraps_openai_errors(tmp_path: Path):
    wav_path = tmp_path / "session.wav"
    wav_path.write_bytes(b"RIFF....WAVEfmt ")

    client = MagicMock()
    client.audio.transcriptions.create.side_effect = APIError(
        message="boom", request=MagicMock(), body=None
    )

    with pytest.raises(TranscriptionError):
        transcribe_audio(wav_path, client=client)


def test_transcribe_audio_raises_when_file_missing(tmp_path: Path):
    missing_path = tmp_path / "missing.wav"
    client = _fake_client("irrelevant")

    with pytest.raises(TranscriptionError):
        transcribe_audio(missing_path, client=client)
