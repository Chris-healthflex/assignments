import os
import wave
from pathlib import Path
from tempfile import NamedTemporaryFile
from fastapi import UploadFile
from app.core.config import get_settings
from app.core.logging import logger
from app.core.errors import AudioValidationError


class AudioValidator:
    """Validates uploaded audio files for format, integrity, and safety constraints."""

    def __init__(self) -> None:
        self.settings = get_settings()

    async def validate_upload(self, file: UploadFile) -> bytes:
        """
        Validates file metadata, reads content, and ensures it conforms to WAV constraints.
        Returns the raw audio bytes if valid.
        """
        if not file or not file.filename:
            logger.warning("Upload rejected: Missing file or filename")
            raise AudioValidationError("Audio file must be provided with a valid filename.")

        filename = file.filename.lower()
        if not any(filename.endswith(ext) for ext in self.settings.allowed_extensions_list):
            logger.warning("Upload rejected: Invalid extension '%s'", file.filename)
            raise AudioValidationError(
                f"Unsupported file format. Expected a .wav file, received: '{file.filename}'"
            )


        # Read file contents
        content = await file.read()
        file_size = len(content)

        if file_size == 0:
            logger.warning("Upload rejected: File is empty")
            raise AudioValidationError("Uploaded audio file is empty (0 bytes).")

        if file_size > self.settings.MAX_AUDIO_SIZE_BYTES:
            max_mb = self.settings.MAX_AUDIO_SIZE_BYTES / (1024 * 1024)
            logger.warning("Upload rejected: Size %d exceeds limit of %d MB", file_size, max_mb)
            raise AudioValidationError(
                f"Audio file size ({file_size / (1024*1024):.2f} MB) exceeds maximum allowed size ({max_mb:.1f} MB)."
            )

        # Validate WAV structure and RIFF header
        self._validate_wav_content(content)

        logger.info("Audio validation successful for '%s' (%d bytes)", file.filename, file_size)
        return content

    def validate_wav_bytes(self, content: bytes, filename: str = "audio.wav") -> None:
        """Validates raw WAV bytes against extension, size, and header constraints."""
        if not filename or not any(filename.lower().endswith(ext) for ext in self.settings.allowed_extensions_list):
            raise AudioValidationError(f"Invalid file extension. Expected .wav, received: '{filename}'")

        if len(content) == 0:
            raise AudioValidationError("Audio file is empty (0 bytes).")

        if len(content) > self.settings.MAX_AUDIO_SIZE_BYTES:
            raise AudioValidationError("Audio file exceeds maximum allowed size.")

        self._validate_wav_content(content)

    def _validate_wav_content(self, content: bytes) -> None:

        """Validates RIFF header and integrity using standard wave module."""
        if len(content) < 44:
            raise AudioValidationError("Audio file is too small to be a valid WAV file.")

        # Check RIFF and WAVE magic bytes
        if not (content[:4] == b"RIFF" and content[8:12] == b"WAVE"):
            raise AudioValidationError("Invalid audio header: File does not appear to be a standard RIFF/WAVE audio file.")

        # Create temporary file to test wave parsing
        temp_file = NamedTemporaryFile(suffix=".wav", delete=False)
        try:
            temp_file.write(content)
            temp_file.flush()
            temp_file.close()

            with wave.open(temp_file.name, "rb") as wav_reader:
                channels = wav_reader.getnchannels()
                sample_width = wav_reader.getsampwidth()
                framerate = wav_reader.getframerate()
                n_frames = wav_reader.getnframes()

                if channels < 1:
                    raise AudioValidationError("Corrupt WAV: Audio has 0 channels.")
                if framerate <= 0:
                    raise AudioValidationError("Corrupt WAV: Invalid sampling rate.")
                if n_frames <= 0:
                    raise AudioValidationError("Corrupt WAV: Audio contains 0 frames.")

                duration_sec = n_frames / float(framerate)
                logger.debug(
                    "WAV verified: %d channels, %d Hz, %d sample width, %.2f seconds",
                    channels, framerate, sample_width, duration_sec
                )
        except wave.Error as exc:
            logger.error("WAV header parsing failed: %s", exc)
            raise AudioValidationError(f"Invalid or corrupted WAV file structure: {str(exc)}")
        except Exception as exc:
            if isinstance(exc, AudioValidationError):
                raise
            logger.error("Unexpected error during WAV parsing: %s", exc)
            raise AudioValidationError(f"Failed to parse audio structure: {str(exc)}")
        finally:
            if os.path.exists(temp_file.name):
                try:
                    os.unlink(temp_file.name)
                except Exception:
                    pass


def get_audio_validator() -> AudioValidator:
    """Dependency provider for AudioValidator."""
    return AudioValidator()
