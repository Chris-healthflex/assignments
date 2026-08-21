from pydub import AudioSegment
from pathlib import Path
import tempfile
import os
import shutil
import imageio_ffmpeg

from app.core.logging import logger

def _ensure_ffmpeg_available():
    """Set pydub's ffmpeg converter to the bundled binary from imageio-ffmpeg."""
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    AudioSegment.converter = ffmpeg_exe
    AudioSegment.ffmpeg = ffmpeg_exe  # some pydub versions need this

def convert_to_16k_mono(input_path: str, output_path: str | None = None) -> str:
    """Convert audio to 16kHz mono PCM WAV."""
    _ensure_ffmpeg_available()
    if output_path is None:
        fd, output_path = tempfile.mkstemp(suffix="_16k.wav")
        os.close(fd)
        Path(output_path).unlink(missing_ok=True)
    try:
        audio = AudioSegment.from_file(input_path)
        audio = audio.set_frame_rate(16000).set_channels(1)
        audio.export(output_path, format="wav", parameters=["-ac", "1", "-ar", "16000"])
        logger.info(f"Audio converted to 16k mono: {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"Audio conversion failed: {e}")
        raise