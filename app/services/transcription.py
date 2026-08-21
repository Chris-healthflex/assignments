from pathlib import Path

from openai import OpenAI, OpenAIError


class TranscriptionError(Exception):
    """Raised when the audio file cannot be transcribed."""


def transcribe_audio(file_path: Path, client: OpenAI | None = None) -> str:
    """Transcribe a WAV file to plain text using OpenAI's Whisper API."""
    client = client or OpenAI()

    try:
        with open(file_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
            )
    except OpenAIError as exc:
        raise TranscriptionError(f"Whisper transcription failed: {exc}") from exc
    except OSError as exc:
        raise TranscriptionError(f"Could not read audio file: {exc}") from exc

    text = transcript.text.strip()
    if not text:
        raise TranscriptionError("Whisper returned an empty transcript")

    return text
