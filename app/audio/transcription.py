from faster_whisper import WhisperModel
from pathlib import Path
from typing import Optional
import threading
import asyncio
from concurrent.futures import ThreadPoolExecutor

from app.core.config import settings
from app.core.logging import logger
from app.audio.preprocessing import convert_to_16k_mono

_model = None
_model_lock = threading.Lock()

def _get_model() -> WhisperModel:
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                logger.info(f"Loading Whisper model: {settings.WHISPER_MODEL}")
                _model = WhisperModel(
                    settings.WHISPER_MODEL,
                    device=settings.WHISPER_DEVICE,
                    compute_type=settings.WHISPER_COMPUTE_TYPE
                )
    return _model

def _transcribe_sync(file_path: str) -> str:
    model = _get_model()
    segments, info = model.transcribe(file_path, beam_size=5, temperature=0.0)
    transcript = " ".join(segment.text for segment in segments).strip()
    return transcript

async def transcribe_wav(file_path: str) -> str:
    """Convert to 16k mono and transcribe using faster-whisper (CPU-bound -> threadpool)."""
    converted_path = await asyncio.to_thread(convert_to_16k_mono, file_path)
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor(max_workers=1) as executor:
        transcript = await loop.run_in_executor(executor, _transcribe_sync, converted_path)
    # cleanup converted file if it's different from original
    if converted_path != file_path:
        Path(converted_path).unlink(missing_ok=True)
    return transcript