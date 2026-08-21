import io
import wave
from pathlib import Path
from unittest.mock import MagicMock

import httpx
import pytest
from groq import APIError, RateLimitError

from app.services import transcription as transcription_module
from app.services.transcription import (
    TranscriptCache,
    TranscriptionError,
    _split_wav,
    transcribe_audio,
)


def _fake_client(text: str, segments: list | None = None) -> MagicMock:
    client = MagicMock()
    client.audio.transcriptions.create.return_value = MagicMock(
        text=text, segments=segments if segments is not None else []
    )
    return client


def _wav_bytes(seconds: float, frame_rate: int = 8000) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(frame_rate)
        handle.writeframes(b"\x00\x00" * int(frame_rate * seconds))
    return buffer.getvalue()


@pytest.fixture
def wav_path(tmp_path: Path) -> Path:
    path = tmp_path / "session.wav"
    path.write_bytes(_wav_bytes(0.1))
    return path


def test_transcribe_audio_returns_stripped_text(wav_path: Path):
    client = _fake_client("  Patient reports lower back pain.  ")

    result = transcribe_audio(wav_path, client=client)

    assert result.text == "Patient reports lower back pain."
    client.audio.transcriptions.create.assert_called_once()


def test_transcribe_audio_parses_timestamped_segments(wav_path: Path):
    client = _fake_client(
        "Where does it hurt? My left knee.",
        segments=[
            {"start": 0.0, "end": 2.5, "text": " Where does it hurt?"},
            {"start": 2.5, "end": 4.0, "text": " My left knee."},
        ],
    )

    result = transcribe_audio(wav_path, client=client)

    assert [segment.id for segment in result.segments] == [0, 1]
    assert result.segments[0].text == "Where does it hurt?"
    assert result.segments[1].start == 2.5
    assert result.duration == 4.0


def test_as_prompt_numbers_segments_for_citation(wav_path: Path):
    client = _fake_client(
        "My left knee.",
        segments=[{"start": 2.5, "end": 4.0, "text": "My left knee."}],
    )

    prompt = transcribe_audio(wav_path, client=client).as_prompt()

    assert prompt == "[0] (2.5s-4.0s) My left knee."


def test_transcribe_audio_falls_back_to_one_segment_without_timestamps(wav_path: Path):
    client = _fake_client("Patient reports pain.")

    result = transcribe_audio(wav_path, client=client)

    assert len(result.segments) == 1
    assert result.segments[0].text == "Patient reports pain."


def test_transcribe_audio_raises_on_empty_transcript(wav_path: Path):
    client = _fake_client("   ")

    with pytest.raises(TranscriptionError):
        transcribe_audio(wav_path, client=client)


def test_transcribe_audio_wraps_groq_errors(wav_path: Path):
    client = MagicMock()
    client.audio.transcriptions.create.side_effect = APIError(
        message="boom", request=MagicMock(), body=None
    )

    with pytest.raises(TranscriptionError):
        transcribe_audio(wav_path, client=client)


def test_transcribe_audio_raises_when_file_missing(tmp_path: Path):
    client = _fake_client("irrelevant")

    with pytest.raises(TranscriptionError):
        transcribe_audio(tmp_path / "missing.wav", client=client)


def test_cache_avoids_transcribing_the_same_audio_twice(wav_path: Path):
    client = _fake_client("Patient reports lower back pain.")
    cache = TranscriptCache()

    first = transcribe_audio(wav_path, client=client, cache=cache)
    second = transcribe_audio(wav_path, client=client, cache=cache)

    assert first.text == second.text
    assert client.audio.transcriptions.create.call_count == 1


def test_cache_misses_when_audio_differs(tmp_path: Path):
    cache = TranscriptCache()
    client = _fake_client("Some text.")

    first = tmp_path / "a.wav"
    first.write_bytes(_wav_bytes(0.1))
    second = tmp_path / "b.wav"
    second.write_bytes(_wav_bytes(0.2))

    transcribe_audio(first, client=client, cache=cache)
    transcribe_audio(second, client=client, cache=cache)

    assert client.audio.transcriptions.create.call_count == 2


def test_retries_rate_limit_errors_then_succeeds(wav_path: Path, monkeypatch):
    monkeypatch.setattr(transcription_module.time, "sleep", lambda _: None)

    rate_limited = RateLimitError(
        "slow down",
        response=httpx.Response(429, request=httpx.Request("POST", "http://groq")),
        body=None,
    )
    client = MagicMock()
    client.audio.transcriptions.create.side_effect = [
        rate_limited,
        MagicMock(text="Recovered transcript.", segments=[]),
    ]

    result = transcribe_audio(wav_path, client=client)

    assert result.text == "Recovered transcript."
    assert client.audio.transcriptions.create.call_count == 2


def test_does_not_retry_client_errors(wav_path: Path, monkeypatch):
    monkeypatch.setattr(transcription_module.time, "sleep", lambda _: None)

    client = MagicMock()
    client.audio.transcriptions.create.side_effect = APIError(
        message="bad file", request=MagicMock(), body=None
    )

    with pytest.raises(TranscriptionError):
        transcribe_audio(wav_path, client=client)

    assert client.audio.transcriptions.create.call_count == 1


def test_split_wav_produces_playable_chunks_with_offsets():
    data = _wav_bytes(seconds=5, frame_rate=8000)

    chunks = _split_wav(data, chunk_seconds=2)

    assert [round(offset, 1) for offset, _ in chunks] == [0.0, 2.0, 4.0]
    for _, payload in chunks:
        with wave.open(io.BytesIO(payload), "rb") as handle:
            assert handle.getframerate() == 8000


def test_split_wav_leaves_short_audio_untouched():
    data = _wav_bytes(seconds=1, frame_rate=8000)

    assert _split_wav(data, chunk_seconds=600) == [(0.0, data)]


def test_long_audio_is_chunked_and_timestamps_are_shifted(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(transcription_module, "MAX_UPLOAD_BYTES", 1000)
    monkeypatch.setattr(transcription_module, "CHUNK_SECONDS", 2)

    path = tmp_path / "long.wav"
    path.write_bytes(_wav_bytes(seconds=4, frame_rate=8000))

    client = MagicMock()
    client.audio.transcriptions.create.side_effect = [
        MagicMock(text="First half.", segments=[{"start": 0.0, "end": 2.0, "text": "First half."}]),
        MagicMock(text="Second half.", segments=[{"start": 0.0, "end": 2.0, "text": "Second half."}]),
    ]

    result = transcribe_audio(path, client=client)

    assert client.audio.transcriptions.create.call_count == 2
    assert result.text == "First half. Second half."
    # Second chunk's timestamps are shifted into whole-file time.
    assert [segment.id for segment in result.segments] == [0, 1]
    assert result.segments[1].start == 2.0
    assert result.segments[1].end == 4.0
