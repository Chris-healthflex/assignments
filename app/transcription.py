import asyncio
import logging
import wave
from pathlib import Path
from typing import Tuple

import numpy as np

from app.config import Settings
from app.errors import PipelineError

logger = logging.getLogger(__name__)

WHISPER_SAMPLE_RATE = 16000
_local_model = None


def load_wav(path: Path) -> Tuple[np.ndarray, float]:
    """Decode a PCM WAV into the float32 mono 16 kHz array Whisper expects.

    Whisper's own loader shells out to ffmpeg. Reading the WAV here keeps the
    service runnable without that system dependency, and rejects unusable
    uploads before a model is loaded.
    """
    try:
        with wave.open(str(path), "rb") as wav:
            channels = wav.getnchannels()
            sample_width = wav.getsampwidth()
            sample_rate = wav.getframerate()
            raw = wav.readframes(wav.getnframes())
    except (wave.Error, EOFError) as exc:
        raise PipelineError(
            "invalid_audio",
            "File is not a readable PCM WAV audio file.",
            400,
            [{"field": "file", "message": str(exc)}],
        ) from exc

    dtypes = {1: np.uint8, 2: np.int16, 4: np.int32}
    if sample_width not in dtypes:
        raise PipelineError(
            "invalid_audio",
            f"Unsupported WAV sample width: {sample_width * 8} bit.",
            400,
            [{"field": "file", "message": "expected 8, 16 or 32 bit PCM"}],
        )
    if not raw:
        raise PipelineError(
            "invalid_audio",
            "WAV file contains no audio frames.",
            400,
            [{"field": "file", "message": "empty audio stream"}],
        )

    samples = np.frombuffer(raw, dtype=dtypes[sample_width])
    if sample_width == 1:
        audio = (samples.astype(np.float32) - 128.0) / 128.0
    else:
        audio = samples.astype(np.float32) / float(np.iinfo(dtypes[sample_width]).max)

    if channels > 1:
        usable = (audio.size // channels) * channels
        audio = audio[:usable].reshape(-1, channels).mean(axis=1)

    duration = audio.size / float(sample_rate)
    if sample_rate != WHISPER_SAMPLE_RATE:
        target = int(round(audio.size * WHISPER_SAMPLE_RATE / sample_rate))
        audio = np.interp(
            np.linspace(0, audio.size - 1, target),
            np.arange(audio.size),
            audio,
        ).astype(np.float32)

    return audio, duration


async def transcribe(path: Path, settings: Settings) -> str:
    audio, duration = load_wav(path)
    if settings.whisper_backend == "openai":
        text = await _transcribe_with_api(path, settings)
    else:
        text = await asyncio.to_thread(_transcribe_locally, audio, settings)

    text = text.strip()
    if not text:
        raise PipelineError(
            "transcription_failed",
            "Whisper returned an empty transcript.",
            502,
            [{"field": "file", "message": "no speech detected in the audio"}],
        )
    logger.info("transcribed %.1fs of audio into %d characters", duration, len(text))
    return text


def _transcribe_locally(audio: np.ndarray, settings: Settings) -> str:
    global _local_model
    try:
        import whisper
    except ImportError as exc:
        raise PipelineError(
            "transcription_failed",
            "The local Whisper backend requires the openai-whisper package.",
            502,
            [{"field": "whisper_backend", "message": str(exc)}],
        ) from exc

    if _local_model is None:
        logger.info("loading whisper model %s", settings.whisper_model)
        _local_model = whisper.load_model(settings.whisper_model)

    try:
        result = _local_model.transcribe(
            audio, language=settings.whisper_language, fp16=False
        )
    except Exception as exc:
        raise PipelineError(
            "transcription_failed",
            "Local Whisper transcription failed.",
            502,
            [{"field": "file", "message": str(exc)}],
        ) from exc
    return result.get("text", "")


async def _transcribe_with_api(path: Path, settings: Settings) -> str:
    if not settings.openai_api_key:
        raise PipelineError(
            "transcription_failed",
            "OPENAI_API_KEY is required for the openai Whisper backend.",
            502,
            [{"field": "openai_api_key", "message": "missing api key"}],
        )
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=settings.openai_api_key)
    try:
        with path.open("rb") as handle:
            result = await client.audio.transcriptions.create(
                model="whisper-1", file=handle, language=settings.whisper_language
            )
    except Exception as exc:
        raise PipelineError(
            "transcription_failed",
            "The OpenAI transcription request failed.",
            502,
            [{"field": "file", "message": str(exc)}],
        ) from exc
    return result.text
