"""Whisper transcription with segment-level timestamps.

The pipeline needs more than plain text: every field the extraction agent
fills has to be traceable back to the moment in the recording it came from.
So we ask Whisper for `verbose_json`, which returns time-coded segments, and
carry those through to the API.

Also handles the two things that bite in production:
  * files over Groq's upload limit are split on WAV frame boundaries and the
    per-chunk timestamps are shifted back into whole-file time;
  * transient rate limits / 5xx / connection drops are retried with backoff.
"""

from __future__ import annotations

import hashlib
import io
import logging
import time
import wave
from pathlib import Path

from groq import (
    APIConnectionError,
    APIStatusError,
    Groq,
    GroqError,
    RateLimitError,
)
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

WHISPER_MODEL = "whisper-large-v3"

# Groq rejects uploads past ~25 MB on the free tier. Stay under it and split
# anything larger rather than failing the request.
MAX_UPLOAD_BYTES = 24 * 1024 * 1024
CHUNK_SECONDS = 600

MAX_ATTEMPTS = 4
BACKOFF_BASE_SECONDS = 1.0


class TranscriptionError(Exception):
    """Raised when the audio file cannot be transcribed."""


class TranscriptSegment(BaseModel):
    """One time-coded slice of the recording."""

    id: int
    start: float
    end: float
    text: str


class Transcript(BaseModel):
    text: str
    segments: list[TranscriptSegment] = Field(default_factory=list)

    @property
    def duration(self) -> float:
        return self.segments[-1].end if self.segments else 0.0

    def as_prompt(self) -> str:
        """Render the transcript with segment ids the agent can cite.

        The agent sees stable `[id]` handles and timestamps, which is what
        makes it possible to ask it *where* each extracted value came from.
        """
        if not self.segments:
            return self.text

        return "\n".join(
            f"[{seg.id}] ({seg.start:.1f}s-{seg.end:.1f}s) {seg.text}"
            for seg in self.segments
        )


class TranscriptCache:
    """Content-addressed transcript cache.

    Re-transcribing a byte-identical WAV is pure waste — the same audio always
    yields the same transcript. Keyed by SHA-256 of the file contents, so it is
    correct across renames and safe to share between requests. Process-local
    and bounded; a real deployment would point this at Redis.
    """

    def __init__(self, max_entries: int = 32):
        self._entries: dict[str, Transcript] = {}
        self._max_entries = max_entries

    @staticmethod
    def key_for(data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

    def get(self, key: str) -> Transcript | None:
        return self._entries.get(key)

    def put(self, key: str, transcript: Transcript) -> None:
        if len(self._entries) >= self._max_entries:
            self._entries.pop(next(iter(self._entries)))
        self._entries[key] = transcript


def _is_retryable(exc: GroqError) -> bool:
    if isinstance(exc, (RateLimitError, APIConnectionError)):
        return True
    # Server-side faults are worth another go; 4xx (bad file, bad key) are not.
    return isinstance(exc, APIStatusError) and exc.status_code >= 500


def _call_whisper(client: Groq, name: str, payload: bytes):
    """Invoke Whisper, retrying only on faults that a retry can actually fix."""
    last_error: GroqError | None = None

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            return client.audio.transcriptions.create(
                model=WHISPER_MODEL,
                file=(name, payload),
                response_format="verbose_json",
            )
        except GroqError as exc:
            last_error = exc
            if not _is_retryable(exc) or attempt == MAX_ATTEMPTS:
                raise
            delay = BACKOFF_BASE_SECONDS * (2 ** (attempt - 1))
            logger.warning(
                "whisper call failed (attempt %d/%d), retrying in %.1fs: %s",
                attempt,
                MAX_ATTEMPTS,
                delay,
                exc,
            )
            time.sleep(delay)

    raise last_error  # unreachable; keeps type checkers happy


def _segments_from_response(response, offset: float, next_id: int) -> list[TranscriptSegment]:
    """Normalise Whisper's segments, shifting them into whole-file time."""
    raw = getattr(response, "segments", None) or []
    segments: list[TranscriptSegment] = []

    for item in raw:
        # The SDK hands back dicts for verbose_json, but be tolerant.
        get = item.get if isinstance(item, dict) else lambda k, d=None: getattr(item, k, d)
        text = (get("text", "") or "").strip()
        if not text:
            continue
        segments.append(
            TranscriptSegment(
                id=next_id + len(segments),
                start=float(get("start", 0.0) or 0.0) + offset,
                end=float(get("end", 0.0) or 0.0) + offset,
                text=text,
            )
        )

    return segments


def _split_wav(data: bytes, chunk_seconds: int | None = None) -> list[tuple[float, bytes]]:
    """Split WAV bytes into (offset_seconds, wav_bytes) on frame boundaries.

    Each chunk is a complete, self-contained WAV file (header included), so
    Whisper can decode it standalone. Returns a single unsplit chunk if the
    file isn't parseable as WAV — the API will surface a clearer error than we
    could invent here.
    """
    chunk_seconds = chunk_seconds if chunk_seconds is not None else CHUNK_SECONDS

    try:
        with wave.open(io.BytesIO(data), "rb") as source:
            params = source.getparams()
            frame_rate = params.framerate
            total_frames = params.nframes
            frames_per_chunk = int(frame_rate * chunk_seconds)

            if frames_per_chunk <= 0 or total_frames <= frames_per_chunk:
                return [(0.0, data)]

            chunks: list[tuple[float, bytes]] = []
            for start_frame in range(0, total_frames, frames_per_chunk):
                source.setpos(start_frame)
                frames = source.readframes(
                    min(frames_per_chunk, total_frames - start_frame)
                )

                buffer = io.BytesIO()
                with wave.open(buffer, "wb") as chunk:
                    chunk.setnchannels(params.nchannels)
                    chunk.setsampwidth(params.sampwidth)
                    chunk.setframerate(frame_rate)
                    chunk.writeframes(frames)

                chunks.append((start_frame / frame_rate, buffer.getvalue()))

            return chunks
    except (wave.Error, EOFError):
        return [(0.0, data)]


def transcribe_audio(
    file_path: Path,
    client: Groq | None = None,
    cache: TranscriptCache | None = None,
) -> Transcript:
    """Transcribe a WAV file into time-coded segments using Whisper."""
    client = client or Groq()

    try:
        data = file_path.read_bytes()
    except OSError as exc:
        raise TranscriptionError(f"Could not read audio file: {exc}") from exc

    cache_key = TranscriptCache.key_for(data) if cache else ""
    if cache:
        cached = cache.get(cache_key)
        if cached is not None:
            logger.info("transcript cache hit for %s", cache_key[:12])
            return cached

    chunks = (
        _split_wav(data) if len(data) > MAX_UPLOAD_BYTES else [(0.0, data)]
    )
    if len(chunks) > 1:
        logger.info("audio split into %d chunks for transcription", len(chunks))

    segments: list[TranscriptSegment] = []
    texts: list[str] = []

    try:
        for offset, payload in chunks:
            response = _call_whisper(client, file_path.name, payload)
            segments.extend(_segments_from_response(response, offset, len(segments)))
            texts.append((getattr(response, "text", "") or "").strip())
    except GroqError as exc:
        raise TranscriptionError(f"Whisper transcription failed: {exc}") from exc

    text = " ".join(part for part in texts if part).strip()
    if not text:
        raise TranscriptionError("Whisper returned an empty transcript")

    # verbose_json normally gives segments; if a provider omits them, degrade to
    # one whole-file segment so evidence linking still has something to point at.
    if not segments:
        segments = [TranscriptSegment(id=0, start=0.0, end=0.0, text=text)]

    transcript = Transcript(text=text, segments=segments)
    if cache:
        cache.put(cache_key, transcript)

    return transcript
