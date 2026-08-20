import abc
import os
import httpx
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any
from openai import OpenAI, OpenAIError
import whisper


from app.core.config import get_settings, Settings
from app.core.logging import logger
from app.core.errors import TranscriptionError


class BaseWhisperTranscriber(abc.ABC):
    """Abstract interface for Whisper audio transcription."""

    @abc.abstractmethod
    async def transcribe(self, audio_bytes: bytes, filename: str = "audio.wav") -> str:
        """Transcribes raw audio bytes to plain text transcript."""
        pass


class OpenAIWhisperTranscriber(BaseWhisperTranscriber):
    """Transcribes audio using OpenAI or Groq Audio API."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        api_key = self.settings.effective_whisper_api_key
        base_url = self.settings.effective_whisper_base_url

        if not api_key:
            logger.warning("No API key configured for OpenAIWhisperTranscriber.")
            self.client = None
        else:
            http_client = httpx.Client(
                headers={"User-Agent": "Mozilla/5.0", "Accept-Encoding": "gzip, deflate"}
            )
            client_kwargs: dict[str, Any] = {
                "api_key": api_key,
                "http_client": http_client,
            }
            if base_url:
                client_kwargs["base_url"] = base_url
                logger.info("OpenAIWhisperTranscriber using base URL: %s", base_url)
            self.client = OpenAI(**client_kwargs)


    async def transcribe(self, audio_bytes: bytes, filename: str = "audio.wav") -> str:
        api_key = self.settings.effective_whisper_api_key
        if not self.client or not api_key:
            raise TranscriptionError(
                "Whisper API key is missing. Please set OPENAI_API_KEY or GROQ_API_KEY or configure WHISPER_MODE=local."
            )

        # Use appropriate model name (whisper-large-v3-turbo for Groq, whisper-1 for OpenAI)
        model = self.settings.WHISPER_MODEL
        if api_key.startswith("gsk_") and model == "whisper-1":
            model = "whisper-large-v3-turbo"

        logger.info("Transcribing audio using Whisper API (model=%s)", model)
        temp_file = NamedTemporaryFile(suffix=".wav", delete=False)
        try:
            temp_file.write(audio_bytes)
            temp_file.flush()
            temp_file.close()

            with open(temp_file.name, "rb") as audio_file:
                response = self.client.audio.transcriptions.create(
                    model=model,
                    file=audio_file,
                    response_format="text"
                )

            transcript = response if isinstance(response, str) else getattr(response, "text", str(response))
            transcript = transcript.strip()

            if not transcript:
                logger.warning("Whisper API returned an empty transcript.")
                raise TranscriptionError("Transcription yielded an empty transcript.")

            logger.info("Whisper API transcription completed successfully (%d characters)", len(transcript))
            return transcript
        except OpenAIError as exc:
            logger.error("Whisper API error: %s", exc)
            raise TranscriptionError(f"Speech-to-text API failed: {str(exc)}")

        except Exception as exc:
            if isinstance(exc, TranscriptionError):
                raise
            logger.error("Unexpected error during OpenAI transcription: %s", exc)
            raise TranscriptionError(f"Transcription failed: {str(exc)}")
        finally:
            if os.path.exists(temp_file.name):
                try:
                    os.unlink(temp_file.name)
                except Exception:
                    pass


class LocalWhisperTranscriber(BaseWhisperTranscriber):
    """Transcribes audio using locally installed OpenAI Whisper model."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._model: Any = None

    def _get_model(self) -> Any:
        if self._model is None:
            logger.info("Loading local Whisper model '%s'...", self.settings.LOCAL_WHISPER_MODEL)
            self._model = whisper.load_model(self.settings.LOCAL_WHISPER_MODEL)
            logger.info("Local Whisper model loaded successfully.")
        return self._model

    async def transcribe(self, audio_bytes: bytes, filename: str = "audio.wav") -> str:
        logger.info("Transcribing audio using Local Whisper model (%s)", self.settings.LOCAL_WHISPER_MODEL)
        temp_file = NamedTemporaryFile(suffix=".wav", delete=False)
        try:
            temp_file.write(audio_bytes)
            temp_file.flush()
            temp_file.close()

            model = self._get_model()
            result = model.transcribe(temp_file.name, fp16=False)
            transcript = result.get("text", "").strip()

            if not transcript:
                logger.warning("Local Whisper returned an empty transcript.")
                raise TranscriptionError("Transcription yielded an empty transcript.")

            logger.info("Local Whisper transcription completed successfully (%d characters)", len(transcript))
            return transcript
        except Exception as exc:
            if isinstance(exc, TranscriptionError):
                raise
            logger.error("Local Whisper transcription failed: %s", exc)
            raise TranscriptionError(f"Local Whisper transcription failed: {str(exc)}")
        finally:
            if os.path.exists(temp_file.name):
                try:
                    os.unlink(temp_file.name)
                except Exception:
                    pass


class MockWhisperTranscriber(BaseWhisperTranscriber):
    """Deterministic mock transcriber for automated testing."""

    def __init__(self, mock_transcript: str | None = None) -> None:
        self.mock_transcript = mock_transcript or (
            "The patient reports knee pain for approximately three weeks after running. "
            "On examination, left knee flexion is 120 degrees and right knee flexion is 135 degrees. "
            "Patient wishes to return to 5k running by October 15. "
            "We recommend physiotherapy sessions twice per week and advised icing for 15 minutes daily."
        )

    async def transcribe(self, audio_bytes: bytes, filename: str = "audio.wav") -> str:
        logger.info("Using MockWhisperTranscriber")
        if not audio_bytes:
            raise TranscriptionError("No audio content provided to mock transcriber.")
        return self.mock_transcript


def get_transcriber() -> BaseWhisperTranscriber:
    """Factory function returning the configured transcriber."""
    settings = get_settings()
    if settings.WHISPER_MODE == "local":
        return LocalWhisperTranscriber(settings)
    elif settings.WHISPER_MODE == "mock":
        return MockWhisperTranscriber()
    else:
        return OpenAIWhisperTranscriber(settings)
