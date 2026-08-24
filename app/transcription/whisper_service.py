"""WAV -> text via local Whisper.

Two backends, selected by config:
  * "faster-whisper" (default): CTranslate2 build, fast on CPU, low memory.
  * "whisper"       : reference openai-whisper.

Both are imported lazily so the rest of the app (schema, API, DB, stub pipeline)
runs and tests pass without the heavy ML deps installed.
"""
from __future__ import annotations

from app.config import settings
from app.transcription.audio_io import validate_wav
from app.transcription.models import Segment, Transcript


class TranscriptionError(RuntimeError):
    pass


def _transcribe_faster_whisper(path: str) -> Transcript:
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:  # pragma: no cover - env dependent
        raise TranscriptionError(
            "faster-whisper is not installed. `pip install faster-whisper` "
            "or set WHISPER_BACKEND=whisper."
        ) from exc

    model = WhisperModel(
        settings.whisper_model,
        device=settings.whisper_device,
        compute_type=settings.whisper_compute_type,
    )
    seg_iter, info = model.transcribe(path, vad_filter=True)
    segments = [Segment(start=s.start, end=s.end, text=s.text.strip()) for s in seg_iter]
    text = " ".join(s.text for s in segments).strip()
    return Transcript(
        text=text,
        language=getattr(info, "language", "en"),
        duration_seconds=getattr(info, "duration", 0.0),
        segments=segments,
        model=settings.whisper_model,
        backend="faster-whisper",
    )


def _transcribe_openai_whisper(path: str) -> Transcript:
    try:
        import whisper
    except ImportError as exc:  # pragma: no cover - env dependent
        raise TranscriptionError(
            "openai-whisper is not installed. `pip install openai-whisper` "
            "or set WHISPER_BACKEND=faster-whisper."
        ) from exc

    model = whisper.load_model(settings.whisper_model, device=settings.whisper_device)
    result = model.transcribe(path)
    segments = [
        Segment(start=float(s["start"]), end=float(s["end"]), text=str(s["text"]).strip())
        for s in result.get("segments", [])
    ]
    return Transcript(
        text=str(result.get("text", "")).strip(),
        language=result.get("language", "en"),
        duration_seconds=segments[-1].end if segments else 0.0,
        segments=segments,
        model=settings.whisper_model,
        backend="whisper",
    )


def transcribe(path: str) -> Transcript:
    """Validate the WAV then transcribe it with the configured backend."""
    info = validate_wav(path)
    if settings.whisper_backend == "whisper":
        tr = _transcribe_openai_whisper(path)
    else:
        tr = _transcribe_faster_whisper(path)
    # Prefer the container-measured duration if the backend didn't report one.
    if not tr.duration_seconds:
        tr.duration_seconds = info.duration_seconds
    return tr
