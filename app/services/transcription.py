"""Audio transcription via Groq's hosted OpenAI Whisper."""

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from groq import Groq

from app.config import (
    TRANSCRIPTION_TIMEOUT_SECONDS,
    WHISPER_MODEL,
    get_groq_api_key,
)
from app.services.prompts import CLINICAL_TRANSCRIPTION_PROMPT

logger = logging.getLogger(__name__)


def _downsample_to_16k(source: Path) -> Optional[Path]:
    """Re-encode audio to 16 kHz mono WAV in a temp file.

    Whisper resamples its input to 16 kHz mono internally, so sending the
    original 44.1 kHz file only inflates the upload. On the provided recording
    this cuts 8.88 MB to 3.22 MB and transcription time from 148s to 60s, with a
    byte-identical transcript.

    Returns None if ffmpeg is unavailable or the conversion fails, in which case
    the caller uploads the original file unchanged. ffmpeg is therefore an
    optional speed-up, not a hard requirement.
    """
    if shutil.which("ffmpeg") is None:
        logger.info("ffmpeg not found; uploading original audio without downsampling.")
        return None

    target = Path(tempfile.gettempdir()) / f"{source.stem}_16k_mono.wav"
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-i", str(source),
                "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
                str(target),
            ],
            check=True,
            capture_output=True,
        )
        logger.info(
            f"Downsampled to 16 kHz mono: "
            f"{source.stat().st_size / 1048576:.2f} MB -> {target.stat().st_size / 1048576:.2f} MB"
        )
        return target
    except Exception as e:
        logger.warning(f"Downsampling failed ({e}); uploading original audio.")
        return None


def transcribe_audio(file_path: str) -> str:
    """Transcribe an audio file using OpenAI Whisper on Groq."""
    audio_path = Path(file_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    groq_key = get_groq_api_key()
    if not groq_key:
        raise ValueError("GROQ_API_KEY environment variable is required for audio transcription.")

    client = Groq(api_key=groq_key, timeout=TRANSCRIPTION_TIMEOUT_SECONDS)
    # large-v3 rather than large-v3-turbo: turbo trades transcription accuracy,
    # and every downstream clinical value depends on the transcript being right.
    logger.info(f"Transcribing {file_path} using Groq Whisper model '{WHISPER_MODEL}'...")

    downsampled = _downsample_to_16k(audio_path)
    upload_path = downsampled or audio_path

    try:
        with open(upload_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                file=(upload_path.name, audio_file.read()),
                model=WHISPER_MODEL,
                language="en",
                prompt=CLINICAL_TRANSCRIPTION_PROMPT,
                temperature=0,
                response_format="text"
            )
        if isinstance(transcription, str):
            return transcription
        return getattr(transcription, "text", str(transcription))
    finally:
        if downsampled is not None:
            downsampled.unlink(missing_ok=True)
