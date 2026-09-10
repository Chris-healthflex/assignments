"""WAV audio to transcript text, without an ffmpeg dependency.

The decode/convert/resample steps are plain array functions kept separate from
the model call: ffmpeg is not installed on the target machine, and Whisper
accepts a float32 waveform directly, so the standard library ``wave`` module is
enough to get there.
"""

import logging
import wave
from pathlib import Path

import numpy as np

from .config import load_settings
from .errors import AudioDecodeError, TranscriptionError

logger = logging.getLogger(__name__)

WHISPER_SAMPLE_RATE = 16_000
SUPPORTED_SAMPLE_WIDTH_BYTES = 2

# Seeds the decoder with terms it would otherwise mis-hear ("range of motion"
# as "range of ocean"). Kept short on purpose: Whisper's prompt window is about
# 224 tokens, and an over-long prompt crowds out the audio context.
CLINICAL_VOCABULARY_PROMPT = (
    "Physiotherapy assessment. Range of motion, ROM, flexion, extension, "
    "abduction, adduction, goniometer, VAS pain score, palpation, "
    "rehabilitation, physiotherapy, bilateral, cervical, lumbar, rotator cuff."
)


def decode_wav(path: Path) -> tuple[np.ndarray, int]:
    """Decode a 16-bit PCM WAV file into mono int16 samples.

    Args:
        path: Path to the WAV file.

    Returns:
        A tuple of the mono sample array (int16) and its sample rate in Hz.

    Raises:
        AudioDecodeError: If the file is not readable 16-bit PCM WAV audio, or
            it contains no frames.
    """
    try:
        with wave.open(str(path), "rb") as wav_file:
            channel_count = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            sample_rate = wav_file.getframerate()
            frames = wav_file.readframes(wav_file.getnframes())
    except (wave.Error, EOFError) as exc:
        raise AudioDecodeError(f"{path.name} is not readable WAV audio: {exc}") from exc

    if sample_width != SUPPORTED_SAMPLE_WIDTH_BYTES:
        raise AudioDecodeError(
            f"{path.name} is {sample_width * 8}-bit audio; only 16-bit PCM WAV "
            "is supported"
        )
    if not frames:
        raise AudioDecodeError(f"{path.name} contains no audio frames")

    # WAV PCM is little-endian by definition, so the byte order is fixed here
    # rather than inherited from the host.
    samples = np.frombuffer(frames, dtype="<i2")
    return _downmix_to_mono(samples, channel_count), sample_rate


def to_float32(samples: np.ndarray) -> np.ndarray:
    """Convert int16 samples to float32 in the range Whisper expects.

    Args:
        samples: int16 sample array.

    Returns:
        A float32 array in [-1.0, 1.0].
    """
    # int16 spans [-32768, 32767]; dividing by 32768 keeps the full negative
    # extreme at exactly -1.0 without ever exceeding the range.
    return samples.astype(np.float32) / 32768.0


def resample(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    """Resample a waveform by linear interpolation.

    Chosen over ``scipy.signal.resample_poly`` to avoid a dependency for a
    single call; Whisper converts the waveform to a mel spectrogram, so the
    difference in interpolation quality is inaudible to it.

    Args:
        samples: Float sample array.
        source_rate: Sample rate of ``samples``, in Hz.
        target_rate: Desired sample rate, in Hz.

    Returns:
        A float32 array at ``target_rate``.

    Raises:
        ValueError: If either rate is not positive.
    """
    if source_rate <= 0 or target_rate <= 0:
        raise ValueError(
            f"sample rates must be positive, got {source_rate} and {target_rate}"
        )
    if source_rate == target_rate or samples.size == 0:
        return samples.astype(np.float32)

    target_length = round(samples.size * target_rate / source_rate)
    source_positions = np.arange(samples.size)
    target_positions = np.linspace(0, samples.size - 1, target_length)
    return np.interp(target_positions, source_positions, samples).astype(np.float32)


def transcribe(path: Path) -> str:
    """Transcribe a WAV recording of a clinical session.

    Args:
        path: Path to the WAV file.

    Returns:
        The transcript text, whitespace-stripped.

    Raises:
        AudioDecodeError: If the file is not readable 16-bit PCM WAV audio.
        TranscriptionError: If the Whisper model fails to load or to decode.
    """
    # Imported here rather than at module scope so the array functions above
    # stay importable and testable without paying torch's import cost.
    import whisper

    samples, sample_rate = decode_wav(path)
    waveform = resample(to_float32(samples), sample_rate, WHISPER_SAMPLE_RATE)

    model_size = load_settings().whisper_model_size
    logger.info("Transcribing %s with Whisper '%s'", path.name, model_size)
    try:
        model = whisper.load_model(model_size)
        result = model.transcribe(
            waveform,
            initial_prompt=CLINICAL_VOCABULARY_PROMPT,
            # This pipeline targets CPU, where fp16 is unsupported and Whisper
            # would fall back to fp32 with a warning anyway.
            fp16=False,
        )
    except Exception as exc:
        # Whisper surfaces load and decode failures as RuntimeError, OSError and
        # download errors alike; the caller only needs "transcription failed",
        # with the original preserved as the cause.
        raise TranscriptionError(f"Whisper failed on {path.name}: {exc}") from exc

    return str(result["text"]).strip()


def _downmix_to_mono(samples: np.ndarray, channel_count: int) -> np.ndarray:
    """Average interleaved channels into a single mono track."""
    if channel_count < 1:
        raise AudioDecodeError(f"WAV reports {channel_count} channels")
    if channel_count == 1:
        return samples
    if samples.size % channel_count:
        raise AudioDecodeError(
            f"WAV frame data is truncated: {samples.size} samples across "
            f"{channel_count} channels"
        )

    # Averaging in int32 first, because summing int16 channels overflows.
    interleaved = samples.reshape(-1, channel_count).astype(np.int32)
    return interleaved.mean(axis=1).astype(np.int16)
