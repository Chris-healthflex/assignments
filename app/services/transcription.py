import os
import logging
from app.core.config import settings

logger = logging.getLogger(__name__)

def validate_wav_file(file_path: str):
    """
    Validates that the file exists, is non-empty, and has a valid WAV signature.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")
        
    if os.path.getsize(file_path) == 0:
        raise ValueError("Audio file is empty")

    # Check WAV header (RIFF ... WAVE)
    try:
        with open(file_path, "rb") as f:
            header = f.read(12)
            if len(header) < 12 or header[0:4] != b"RIFF" or header[8:12] != b"WAVE":
                raise ValueError("File is not a valid WAV audio file")
    except Exception as e:
        if isinstance(e, ValueError):
            raise
        raise ValueError(f"Could not read audio file header: {e}")

def transcribe_audio(file_path: str) -> str:
    """
    Transcribes a WAV audio file to text.
    Uses Groq's API by default, or falls back to local faster-whisper.
    """
    validate_wav_file(file_path)

    provider = settings.whisper_provider.lower().strip()
    logger.info(f"Starting audio transcription using provider: {provider}")

    if provider == "groq":
        if not settings.groq_api_key:
            raise ValueError("GROQ_API_KEY must be set to use the Groq transcription provider.")
            
        try:
            from groq import Groq
            api_key = settings.groq_api_key.strip('"').strip("'")
            base_url = settings.groq_base_url.strip('"').strip("'")
            
            if "api.groq.com" in base_url:
                client = Groq(api_key=api_key)
            else:
                client = Groq(api_key=api_key, base_url=base_url)

            
            with open(file_path, "rb") as f:

                response = client.audio.transcriptions.create(
                    file=(os.path.basename(file_path), f.read()),
                    model=settings.whisper_model,
                    response_format="json"
                )
            
            transcript_text = response.text.strip()
            if not transcript_text:
                raise ValueError("Transcribed text is empty")
            return transcript_text
            
        except Exception as e:
            logger.error(f"Groq audio transcription failed: {e}")
            raise RuntimeError(f"Groq transcription failed: {e}") from e

    elif provider == "local":
        try:
            from faster_whisper import WhisperModel
        except ImportError as e:
            logger.error("faster-whisper is not installed. To use the local provider, install faster-whisper.")
            raise ImportError(
                "faster-whisper is not installed. Please install it to use the local transcription provider."
            ) from e
            
        try:
            logger.info(f"Loading local Whisper model: {settings.whisper_model}")
            model = WhisperModel(
                settings.whisper_model,
                device=settings.whisper_device,
                compute_type=settings.whisper_compute_type
            )
            segments, info = model.transcribe(file_path, beam_size=5)
            transcript_text = " ".join([segment.text for segment in segments]).strip()
            
            if not transcript_text:
                raise ValueError("Local transcription returned empty text")
            return transcript_text
            
        except Exception as e:
            logger.error(f"Local audio transcription failed: {e}")
            raise RuntimeError(f"Local transcription failed: {e}") from e
    else:
        raise ValueError(f"Unsupported Whisper provider: {provider}")
