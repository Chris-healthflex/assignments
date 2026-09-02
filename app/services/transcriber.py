"""Whisper audio transcription service.

Converts WAV audio recordings into plain text transcripts using OpenAI Whisper API.
"""

from pathlib import Path
from typing import Optional, Set, Union
from openai import OpenAI, OpenAIError

from app.config import settings

# Supported audio file extensions
SUPPORTED_AUDIO_EXTENSIONS: Set[str] = {
    ".wav",
    ".mp3",
    ".m4a",
    ".ogg",
    ".flac",
    ".webm",
}


class TranscriberException(Exception):
    """Base exception for transcription and audio validation errors."""

    pass


class AudioValidationError(TranscriberException):
    """Raised when an audio file is missing, empty, or has an invalid format."""

    pass


class TranscriptionError(TranscriberException):
    """Raised when audio transcription fails."""

    pass


class WhisperTranscriber:
    """Service for transcribing audio recordings via OpenAI Whisper."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        client: Optional[OpenAI] = None,
    ) -> None:
        """Initialize WhisperTranscriber with settings or injected dependencies.

        Args:
            api_key: Optional OpenAI API key override.
            model: Optional Whisper model override (e.g. 'whisper-1').
            client: Optional pre-configured OpenAI client (useful for unit testing).
        """
        self._api_key = api_key if api_key is not None else settings.OPENAI_API_KEY
        self.model = model or settings.WHISPER_MODEL
        self._client = client

    @property
    def client(self) -> OpenAI:
        """Get or lazily initialize OpenAI client."""
        if self._client is not None:
            return self._client

        if not self._api_key or self._api_key.strip() in {
            "",
            "your_openai_api_key_here",
            "mock_key",
        }:
            raise TranscriptionError(
                "OpenAI API key is not configured. Please set OPENAI_API_KEY in environment or .env."
            )

        self._client = OpenAI(api_key=self._api_key)
        return self._client

    def validate_audio_file(self, audio_path: Union[str, Path]) -> Path:
        """Validate that the audio file exists, is non-empty, and has a supported format.

        Args:
            audio_path: Path to the audio file.

        Returns:
            Resolved Path object.

        Raises:
            AudioValidationError: If file does not exist, is empty, or is unsupported.
        """
        path = Path(audio_path)

        if not path.exists():
            raise AudioValidationError(f"Audio file not found: {path}")

        if not path.is_file():
            raise AudioValidationError(f"Specified path is not a file: {path}")

        if path.stat().st_size == 0:
            raise AudioValidationError(f"Audio file is empty (0 bytes): {path.name}")

        ext = path.suffix.lower()
        if ext not in SUPPORTED_AUDIO_EXTENSIONS:
            raise AudioValidationError(
                f"Unsupported audio format '{ext}'. Supported formats: {', '.join(sorted(SUPPORTED_AUDIO_EXTENSIONS))}"
            )

        return path

    def transcribe(self, audio_path: Union[str, Path]) -> str:
        """Transcribe an audio file into plain text.

        Args:
            audio_path: Path to the audio file.

        Returns:
            Plain text transcript.

        Raises:
            AudioValidationError: If audio validation fails.
            TranscriptionError: If the transcription API call fails.
        """
        validated_path = self.validate_audio_file(audio_path)

        try:
            with open(validated_path, "rb") as audio_file:
                response = self.client.audio.transcriptions.create(
                    model=self.model,
                    file=audio_file,
                )

            # Response object has .text attribute
            if hasattr(response, "text"):
                transcript = response.text
            elif isinstance(response, dict) and "text" in response:
                transcript = response["text"]
            else:
                transcript = str(response)

            cleaned_transcript = transcript.strip()
            if not cleaned_transcript:
                raise TranscriptionError("Transcription returned an empty transcript.")

            return cleaned_transcript

        except AudioValidationError:
            raise
        except TranscriptionError:
            raise
        except OpenAIError as exc:
            # Wrap OpenAI API errors without exposing credentials or internal traces
            raise TranscriptionError(f"Whisper API transcription error: {str(exc)}") from exc
        except Exception as exc:
            raise TranscriptionError(f"Failed to process audio file: {str(exc)}") from exc


def transcribe_audio(
    audio_path: Union[str, Path],
    transcriber: Optional[WhisperTranscriber] = None,
) -> str:
    """Convenience function to transcribe audio using WhisperTranscriber.

    Args:
        audio_path: Path to the audio file.
        transcriber: Optional custom WhisperTranscriber instance.

    Returns:
        Plain text transcript.
    """
    service = transcriber or WhisperTranscriber()
    return service.transcribe(audio_path)
