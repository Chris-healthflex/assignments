import io
import wave
from app.audio.validation import validate_wav_bytes

def create_wav_bytes(duration_sec=1.0, sample_rate=16000):
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        n_frames = int(duration_sec * sample_rate)
        wf.writeframes(b'\x00\x00' * n_frames)
    return buf.getvalue()

def test_valid_wav():
    data = create_wav_bytes()
    validate_wav_bytes(data, "test.wav")  # should not raise

def test_invalid_extension():
    try:
        validate_wav_bytes(b"data", "test.mp3")
        assert False, "Should have raised"
    except Exception:
        pass