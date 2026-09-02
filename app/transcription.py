"""
WAV -> text using OpenAI Whisper.

Two backends, selected by WHISPER_BACKEND:
  * local : openai-whisper running on this machine (default, no API cost)
  * api   : OpenAI hosted `whisper-1`

The local path decodes the WAV with `soundfile` and resamples to 16 kHz mono
with scipy, so ffmpeg is NOT required (whisper.load_audio would need it).
"""
from __future__ import annotations

import io
import logging
from functools import lru_cache

import numpy as np

from .config import settings

log = logging.getLogger(__name__)

WHISPER_SR = 16_000


def _load_wav_16k_mono(wav_bytes: bytes) -> np.ndarray:
    import soundfile as sf
    from scipy.signal import resample_poly
    from math import gcd

    audio, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)  # downmix to mono
    if sr != WHISPER_SR:
        g = gcd(sr, WHISPER_SR)
        audio = resample_poly(audio, WHISPER_SR // g, sr // g).astype(np.float32)
    return audio


@lru_cache(maxsize=1)
def _local_model():
    import whisper  # openai-whisper

    log.info("Loading local Whisper model '%s' (first call downloads weights)", settings.whisper_model)
    return whisper.load_model(settings.whisper_model)


def _transcribe_local(wav_bytes: bytes) -> str:
    audio = _load_wav_16k_mono(wav_bytes)
    result = _local_model().transcribe(
        audio,
        language="en",
        fp16=False,
        # Priming the decoder with domain vocabulary markedly improves clinical
        # term recognition (e.g. "patellar mobility" instead of "tele-mobility")
        # and preserves signed ROM values such as "negative 5 degrees".
        initial_prompt=settings.whisper_initial_prompt or None,
    )
    return result["text"].strip()


def _transcribe_api(wav_bytes: bytes) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=settings.openai_api_key)
    f = io.BytesIO(wav_bytes)
    f.name = "audio.wav"
    result = client.audio.transcriptions.create(
        model="whisper-1", file=f, language="en",
        prompt=settings.whisper_initial_prompt or None,
    )
    return result.text.strip()


def transcribe(wav_bytes: bytes) -> str:
    """Return the plain-text transcript of a WAV byte string."""
    if settings.whisper_backend == "api":
        return _transcribe_api(wav_bytes)
    return _transcribe_local(wav_bytes)
