import wave

import numpy as np
import pytest

from app.errors import PipelineError
from app.transcription import WHISPER_SAMPLE_RATE, load_wav


def write_wav(path, sample_rate=44100, channels=1, seconds=1.0):
    frames = np.zeros(int(sample_rate * seconds * channels), dtype=np.int16)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(frames.tobytes())
    return path


def test_a_wav_is_downmixed_and_resampled_for_whisper(tmp_path):
    path = write_wav(tmp_path / "stereo.wav", channels=2)

    audio, duration = load_wav(path)

    assert audio.dtype == np.float32
    assert audio.ndim == 1
    assert duration == pytest.approx(1.0, abs=0.01)
    assert audio.size == pytest.approx(WHISPER_SAMPLE_RATE, abs=2)


def test_the_provided_recording_is_readable(wav_path):
    audio, duration = load_wav(wav_path)

    assert duration > 1.0
    assert audio.size == pytest.approx(duration * WHISPER_SAMPLE_RATE, rel=0.01)


def test_a_file_that_is_not_wav_audio_is_rejected(tmp_path):
    path = tmp_path / "note.wav"
    path.write_bytes(b"this is not audio")

    with pytest.raises(PipelineError) as excinfo:
        load_wav(path)

    assert excinfo.value.code == "invalid_audio"
    assert excinfo.value.status_code == 400


def test_a_wav_without_audio_frames_is_rejected(tmp_path):
    path = write_wav(tmp_path / "silent.wav", seconds=0)

    with pytest.raises(PipelineError) as excinfo:
        load_wav(path)

    assert excinfo.value.details[0]["field"] == "file"
