"""Local Whisper transcription.

Uses openai-whisper directly rather than the OpenAI API so the pipeline has
no hard dependency on an external API for the transcription step (the brief
allows "local or API" — local keeps this fully self-contained and free to
run against the sample WAV repeatedly while iterating on the extraction
agent).
"""
from __future__ import annotations

import functools
import logging

import whisper

from app.config import get_settings

logger = logging.getLogger(__name__)


@functools.lru_cache(maxsize=1)
def _load_model():
    settings = get_settings()
    logger.info("Loading Whisper model=%s device=%s", settings.whisper_model, settings.whisper_device)
    return whisper.load_model(settings.whisper_model, device=settings.whisper_device)


class TranscriptionResult:
    def __init__(self, text: str, segments: list[dict], language: str):
        self.text = text.strip()
        self.segments = segments
        self.language = language


def transcribe_wav(file_path: str) -> TranscriptionResult:
    """Transcribe a WAV file on disk and return text + segment-level detail.

    Segment-level timestamps/avg_logprob are kept because the extraction
    agent uses low per-segment confidence as one signal when deciding
    whether a downstream clinical field should be flagged rather than
    trusted outright.
    """
    model = _load_model()
    result = model.transcribe(file_path, fp16=False, verbose=False)

    segments = [
        {
            "id": seg["id"],
            "start": seg["start"],
            "end": seg["end"],
            "text": seg["text"].strip(),
            "avg_logprob": seg.get("avg_logprob"),
            "no_speech_prob": seg.get("no_speech_prob"),
        }
        for seg in result.get("segments", [])
    ]

    return TranscriptionResult(
        text=result["text"],
        segments=segments,
        language=result.get("language", "en"),
    )
