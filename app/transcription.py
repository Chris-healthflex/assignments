from __future__ import annotations

import io
import math
import logging
from typing import Optional, Union
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from app.config import settings

logger = logging.getLogger(__name__)

# Whisper model cache so we don't reload on every request
_whisper_model = None

CLINICAL_PROMPT = (
    "Clinical physiotherapy initial assessment session discussing patient history, "
    "chief complaint, pain levels, range of motion, flexion, extension degrees, "
    "strength, functional goals, treatment frequency, and patient advice."
)


def get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        import whisper
        logger.info(f"Loading Whisper model '{settings.WHISPER_MODEL}'...")
        _whisper_model = whisper.load_model(settings.WHISPER_MODEL)
    return _whisper_model


def decode_wav_to_16k_mono(audio_bytes: bytes) -> np.ndarray:
    """
    Decodes in-memory WAV audio bytes to 16 kHz single-channel float32 numpy array
    without requiring external ffmpeg installation.
    """
    if not audio_bytes or len(audio_bytes) < 44:  # standard WAV header is at least 44 bytes
        raise ValueError("Audio file is empty or too small to be a valid WAV.")

    try:
        with io.BytesIO(audio_bytes) as bio:
            data, sr = sf.read(bio, dtype="float32")
    except Exception as e:
        raise ValueError(f"Could not decode audio as WAV: {str(e)}") from e

    # Convert stereo / multi-channel to mono
    if data.ndim > 1:
        data = data.mean(axis=1)

    # Resample to 16000 Hz if needed
    if sr != 16000:
        gcd = math.gcd(sr, 16000)
        up = 16000 // gcd
        down = sr // gcd
        data = resample_poly(data, up, down).astype(np.float32)

    return data


def transcribe_audio(audio_data: Union[bytes, str], language: str = "en") -> str:
    """
    Transcribes audio (either raw bytes or path to file).
    Supports local Whisper model and OpenAI Whisper API.
    """
    if isinstance(audio_data, str):
        with open(audio_data, "rb") as f:
            audio_bytes = f.read()
    else:
        audio_bytes = audio_data

    # Check file size limit
    max_bytes = settings.MAX_AUDIO_SIZE_MB * 1024 * 1024
    if len(audio_bytes) > max_bytes:
        raise ValueError(f"Audio file size exceeds limit of {settings.MAX_AUDIO_SIZE_MB}MB.")

    # API Backend
    if settings.WHISPER_BACKEND == "api":
        if not settings.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is required when WHISPER_BACKEND='api'.")
        from openai import OpenAI
        client = OpenAI(api_key=settings.OPENAI_API_KEY)
        bio = io.BytesIO(audio_bytes)
        bio.name = "audio.wav"
        res = client.audio.transcriptions.create(
            model="whisper-1",
            file=bio,
            language=language,
            prompt=CLINICAL_PROMPT,
        )
        return res.text.strip()

    # Local Whisper Backend (Default)
    audio_array = decode_wav_to_16k_mono(audio_bytes)
    model = get_whisper_model()
    result = model.transcribe(
        audio_array,
        language=language or settings.WHISPER_LANGUAGE,
        initial_prompt=CLINICAL_PROMPT,
        fp16=False,  # CPU compatibility
    )
    return result["text"].strip()
