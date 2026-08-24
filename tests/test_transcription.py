"""Transcription service surface (does not require Whisper installed)."""
from app.transcription.models import Transcript, Segment


def test_transcript_meta_shape():
    tr = Transcript(text="hello", language="en", duration_seconds=1.23,
                    segments=[Segment(0, 1, "hello")], model="small", backend="faster-whisper")
    meta = tr.as_meta()
    assert meta["segments"] == 1
    assert meta["durationSeconds"] == 1.23
    assert meta["backend"] == "faster-whisper"


def test_transcribe_import_is_lazy():
    # importing the service must not require faster-whisper to be installed
    import app.transcription.whisper_service as ws
    assert hasattr(ws, "transcribe")
