from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Union, BinaryIO, Optional
import numpy as np

from app.services.audio import load_and_resample_wav

_WHISPER_MODELS = {}


@dataclass
class TranscriptResult:
    text: str
    segments: List[Tuple[float, float, str]]  # (start_sec, end_sec, text)


def get_whisper_model(model_size: str = "base"):
    """Caches loaded Whisper model instances in memory."""
    import whisper

    if model_size not in _WHISPER_MODELS:
        _WHISPER_MODELS[model_size] = whisper.load_model(model_size)
    return _WHISPER_MODELS[model_size]


def transcribe(
    audio_source: Union[str, Path, bytes, BinaryIO, np.ndarray],
    *,
    model_size: str = "base",
    initial_prompt: Optional[str] = None,
) -> TranscriptResult:
    """
    Transcribes audio into structured text and timestamped segments.
    Uses ffmpeg-free soundfile+scipy decoding if given a file/stream.
    """
    if isinstance(audio_source, np.ndarray):
        audio_array = audio_source
    else:
        audio_array = load_and_resample_wav(audio_source, target_sr=16000)

    model = get_whisper_model(model_size)

    prompt = initial_prompt or (
        "Clinical assessment, physical therapy, range of motion, ROM, degrees, knee flexion, "
        "extension, pain scale, strength, VAS, bilateral, degrees of movement."
    )

    result = model.transcribe(
        audio_array,
        language="en",
        fp16=False,
        initial_prompt=prompt,
        verbose=False,
    )

    full_text = result.get("text", "").strip()
    raw_segments = result.get("segments", [])

    segments: List[Tuple[float, float, str]] = []
    for s in raw_segments:
        start = float(s.get("start", 0.0))
        end = float(s.get("end", 0.0))
        text = s.get("text", "").strip()
        segments.append((start, end, text))

    return TranscriptResult(text=full_text, segments=segments)
