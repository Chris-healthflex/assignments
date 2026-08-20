"""Whisper transcription (D2): WAV to text.

Two interchangeable backends run the same OpenAI Whisper weights:

* ``faster`` - faster-whisper / CTranslate2. The default. Needs no torch and
  no ffmpeg, and runs roughly 4x faster on CPU.
* ``openai`` - the literal ``openai-whisper`` package, for reviewers who want
  the reference implementation. Requires requirements-optional.txt.

Both receive an identical numpy array from :mod:`app.transcription.audio_io`,
so switching backends cannot change how the audio was decoded.

Whisper deliberately runs on the CPU: the GPU's 4 GB of VRAM is reserved for
the extraction model, and the two must never compete for it.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

from pydantic import BaseModel, Field

from app.config import Settings, get_settings
from app.transcription.audio_io import load_wav_16k_mono

logger = logging.getLogger(__name__)


class TranscriptSegment(BaseModel):
    """One utterance-sized chunk, with timings kept for clinician review."""

    start: float
    end: float
    text: str


class Transcript(BaseModel):
    text: str
    segments: list[TranscriptSegment] = Field(default_factory=list)
    language: str = ""
    durationSeconds: float = 0.0
    backend: str = ""
    model: str = ""
    transcribeSeconds: float = 0.0


class TranscriptionError(RuntimeError):
    """A Whisper backend is unavailable or misconfigured.

    This is a server-side fault and surfaces as HTTP 503.
    """


class EmptyTranscriptError(TranscriptionError):
    """The audio decoded fine but contains no speech.

    Deliberately distinct from its parent: the service is working, the upload
    is unusable. Reporting that as 503 sends the caller looking for an outage
    when what they need to do is re-record.
    """


# --------------------------------------------------------------------------
# Backends
# --------------------------------------------------------------------------
class _FasterWhisperBackend:
    name = "faster-whisper"

    def __init__(self, settings: Settings) -> None:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:  # pragma: no cover - required dependency
            raise TranscriptionError(
                "faster-whisper is not installed. Run: pip install -r requirements.txt"
            ) from exc

        logger.info("Loading faster-whisper model %r", settings.whisper_model)
        self._model = WhisperModel(
            settings.whisper_model,
            device=settings.whisper_device,
            compute_type=settings.whisper_compute_type,
        )
        self._settings = settings

    def run(self, audio) -> tuple[list[TranscriptSegment], str]:
        segments, info = self._model.transcribe(
            audio,
            language=self._settings.whisper_language or None,
            beam_size=self._settings.whisper_beam_size,
            # VAD trims silence, which stops Whisper hallucinating filler text
            # into the gaps between clinician and patient turns.
            vad_filter=True,
            # Each segment is transcribed independently. Conditioning on prior
            # text makes Whisper repeat itself when the audio is unclear, and a
            # repeated clinical phrase is worse than a missing one.
            condition_on_previous_text=False,
        )
        # `segments` is a generator: consuming it is what runs the model.
        out = [
            TranscriptSegment(start=s.start, end=s.end, text=s.text.strip())
            for s in segments
        ]
        return out, (info.language or "")


class _OpenAIWhisperBackend:
    name = "openai-whisper"

    def __init__(self, settings: Settings) -> None:
        try:
            import whisper
        except ImportError as exc:
            raise TranscriptionError(
                "openai-whisper is not installed. Run: "
                "pip install -r requirements-optional.txt, or set WHISPER_BACKEND=faster"
            ) from exc

        logger.info("Loading openai-whisper model %r", settings.whisper_model)
        self._model = whisper.load_model(
            settings.whisper_model, device=settings.whisper_device
        )
        self._settings = settings

    def run(self, audio) -> tuple[list[TranscriptSegment], str]:
        result = self._model.transcribe(
            audio,
            language=self._settings.whisper_language or None,
            fp16=False,                      # CPU has no half-precision path
            condition_on_previous_text=False,
        )
        out = [
            TranscriptSegment(
                start=float(s["start"]), end=float(s["end"]), text=s["text"].strip()
            )
            for s in result.get("segments", [])
        ]
        return out, result.get("language", "")


_BACKENDS = {"faster": _FasterWhisperBackend, "openai": _OpenAIWhisperBackend}


class WhisperTranscriber:
    """Thread-safe wrapper around a single loaded Whisper model."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._backend = None
        self._lock = threading.Lock()

    def _ensure_backend(self):
        # Double-checked locking: constructing the model is slow, but we do not
        # want to hold a lock on that path once it is built.
        if self._backend is None:
            with self._lock:
                if self._backend is None:
                    backend_cls = _BACKENDS[self._settings.whisper_backend]
                    self._backend = backend_cls(self._settings)
        return self._backend

    def preload(self) -> None:
        """Load the model up front, so the first request is not slow."""
        self._ensure_backend()

    def transcribe(self, wav_path: str | Path) -> Transcript:
        """Decode a WAV file and return its transcript."""
        audio, duration = load_wav_16k_mono(wav_path)
        backend = self._ensure_backend()

        started = time.perf_counter()
        # CTranslate2 models are not safe for concurrent calls.
        with self._lock:
            segments, language = backend.run(audio)
        elapsed = time.perf_counter() - started

        text = " ".join(s.text for s in segments if s.text).strip()
        if not text:
            raise EmptyTranscriptError(
                "No speech was found in this recording. It decoded correctly "
                f"({duration:.0f} seconds of audio), but Whisper produced no "
                "words - check that the right file was uploaded and that the "
                "microphone was recording."
            )

        logger.info(
            "Transcribed %.1fs of audio in %.1fs (%d segments)",
            duration, elapsed, len(segments),
        )
        return Transcript(
            text=text,
            segments=segments,
            language=language,
            durationSeconds=round(duration, 2),
            backend=backend.name,
            model=self._settings.whisper_model,
            transcribeSeconds=round(elapsed, 2),
        )


_transcriber: WhisperTranscriber | None = None
_transcriber_lock = threading.Lock()


def get_transcriber() -> WhisperTranscriber:
    """Process-wide transcriber, so the model loads once per server."""
    global _transcriber
    if _transcriber is None:
        with _transcriber_lock:
            if _transcriber is None:
                _transcriber = WhisperTranscriber()
    return _transcriber
