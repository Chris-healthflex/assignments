from __future__ import annotations
import logging
import shutil
import subprocess
import tempfile
import wave
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)

GROQ_BASE_URL = "https://api.groq.com/openai/v1"


class TranscriptionError(RuntimeError):
    """Raised when a transcription request fails or produces no text."""
@dataclass
class TranscriptSegment:
    start: float
    end: float
    text: str


@dataclass
class Transcript:
    text: str
    language: str = ""
    duration: float = 0.0
    segments: list[TranscriptSegment] = field(default_factory=list)


def probe_wav(path: Path) -> float:
    try:
        with wave.open(str(path), "rb") as wf:
            frames, rate = wf.getnframes(), wf.getframerate()
            if rate <= 0:
                raise TranscriptionError("WAV file reports a zero sample rate.")
            return frames / float(rate)
    except wave.Error as exc:
        raise TranscriptionError(f"Not a readable WAV file: {exc}") from exc


@lru_cache(maxsize=2)
def _load_local_model(model: str, device: str, compute_type: str):
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:  # optional extra
        raise TranscriptionError(
            "WHISPER_BACKEND=local needs the local-whisper extra: "
            "`uv sync --extra local-whisper`."
        ) from exc

    logger.info("Loading faster-whisper model=%s device=%s", model, device)
    return WhisperModel(model, device=device, compute_type=compute_type)


def _transcribe_local(path: Path, settings: Settings) -> Transcript:
    model = _load_local_model(
        settings.whisper_model, settings.whisper_device, settings.whisper_compute_type
    )
    segments, info = model.transcribe(
        str(path),
        language=settings.whisper_language,
        vad_filter=True,         
        beam_size=5,
        condition_on_previous_text=False,
    )
    collected: list[TranscriptSegment] = []
    for seg in segments:  
        text = (seg.text or "").strip()
        if text:
            collected.append(TranscriptSegment(seg.start, seg.end, text))

    return Transcript(
        text=" ".join(s.text for s in collected).strip(),
        language=getattr(info, "language", "") or "",
        duration=float(getattr(info, "duration", 0.0) or 0.0),
        segments=collected,
    )


def _downsample_16k_mono(path: Path) -> tuple[Path, tempfile.TemporaryDirectory | None]:
    if shutil.which("ffmpeg") is None:
        logger.warning("ffmpeg not found; uploading original audio without downsampling.")
        return path, None

    holder = tempfile.TemporaryDirectory(prefix="whisper-16k-")
    out = Path(holder.name) / "audio16k.wav"
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-i", str(path), "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(out)],
        capture_output=True, text=True,
    )
    if proc.returncode != 0 or not out.is_file():
        logger.warning("ffmpeg downsample failed (%s); using original audio.",
                       (proc.stderr or "").strip()[:200])
        holder.cleanup()
        return path, None

    logger.info("Downsampled %s -> %s bytes for upload",
                path.stat().st_size, out.stat().st_size)
    return out, holder


def _transcribe_hosted(
    path: Path, settings: Settings, *, base_url: str | None, api_key: str, model: str
) -> Transcript:
    from openai import OpenAI

    upload_path, holder = (
        _downsample_16k_mono(path) if settings.whisper_downsample else (path, None)
    )
    try:
        size = upload_path.stat().st_size
        if size > settings.whisper_max_request_bytes:
            raise TranscriptionError(
                f"Audio is {size} bytes after preprocessing, over the "
                f"{settings.whisper_max_request_bytes} byte request limit. "
                "Split the recording, or raise WHISPER_MAX_REQUEST_BYTES if your "
                "plan allows a larger payload."
            )

        client = OpenAI(api_key=api_key, base_url=base_url, timeout=300, max_retries=2)
        with upload_path.open("rb") as fh:
            result = client.audio.transcriptions.create(
                model=model,
                file=fh,
                response_format="verbose_json",
                language=settings.whisper_language or None,
            )
    finally:
        if holder is not None:
            holder.cleanup()

    segments: list[TranscriptSegment] = []
    for raw in getattr(result, "segments", None) or []:
        get = raw.get if isinstance(raw, dict) else lambda k, d=None: getattr(raw, k, d)
        text = (get("text") or "").strip()
        if text:
            segments.append(
                TranscriptSegment(float(get("start", 0.0) or 0.0),
                                  float(get("end", 0.0) or 0.0), text)
            )

    return Transcript(
        text=(getattr(result, "text", "") or "").strip(),
        language=getattr(result, "language", "") or "",
        duration=float(getattr(result, "duration", 0.0) or 0.0),
        segments=segments,
    )


def _transcribe_groq(path: Path, settings: Settings) -> Transcript:
    if not settings.groq_api_key:
        raise TranscriptionError(
            "WHISPER_BACKEND=groq requires GROQ_API_KEY to be set in .env."
        )
    return _transcribe_hosted(
        path, settings,
        base_url=GROQ_BASE_URL,
        api_key=settings.groq_api_key,
        model=settings.whisper_model,
    )


def _transcribe_openai(path: Path, settings: Settings) -> Transcript:
    if not settings.openai_api_key:
        raise TranscriptionError(
            "WHISPER_BACKEND=openai requires OPENAI_API_KEY to be set."
        )
    model = settings.whisper_model
    if model.startswith("whisper-large"):
        model = "whisper-1" 
    return _transcribe_hosted(
        path, settings, base_url=None, api_key=settings.openai_api_key, model=model
    )


def transcribe(path: Path, settings: Settings | None = None) -> Transcript:
    """Transcribe a WAV file with the configured backend."""
    settings = settings or get_settings()
    duration = probe_wav(path)

    if settings.whisper_backend == "groq":
        transcript = _transcribe_groq(path, settings)
    elif settings.whisper_backend == "openai":
        transcript = _transcribe_openai(path, settings)
    else:
        transcript = _transcribe_local(path, settings)

    if not transcript.duration:
        transcript.duration = duration
    if not transcript.text:
        raise TranscriptionError(
            "Whisper produced an empty transcript — the audio may be silent."
        )
    return transcript

