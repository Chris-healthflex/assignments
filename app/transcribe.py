"""Whisper transcription adapter."""
from __future__ import annotations

import os
from pathlib import Path


class TranscriptionError(RuntimeError):
    pass


def transcribe_wav(audio_path: Path) -> str:
    """Transcribe a WAV using OpenAI's transcription API.

    The import is deliberately local so schema/API tests do not require the
    optional runtime client to be initialised.
    """
    if not audio_path.is_file():
        raise TranscriptionError("Audio file was not found.")
    if audio_path.suffix.lower() != ".wav":
        raise TranscriptionError("Only WAV audio is supported.")
    if not os.getenv("OPENAI_API_KEY"):
        raise TranscriptionError("OPENAI_API_KEY is required for transcription.")

    try:
        from openai import OpenAI

        client = OpenAI()
        with audio_path.open("rb") as audio:
            response = client.audio.transcriptions.create(
                model=os.getenv("WHISPER_MODEL", "whisper-1"), file=audio
            )
        text = response.text.strip()
    except Exception as exc:  # provider errors must not crash the API
        raise TranscriptionError(f"Whisper transcription failed: {exc}") from exc
    if not text:
        raise TranscriptionError("Whisper returned an empty transcript.")
    return text
