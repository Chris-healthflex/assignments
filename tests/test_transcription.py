import os
import io
import pytest
from app.services.transcription import TranscriptionService
from tests.generate_sample_audio import generate_clinical_wav


def test_wav_validation(tmp_path):
    wav_path = str(tmp_path / "test.wav")
    generate_clinical_wav(wav_path, duration_sec=1.0)
    
    with open(wav_path, "rb") as f:
        audio_bytes = f.read()

    assert TranscriptionService.validate_wav(audio_bytes) is True
    assert TranscriptionService.validate_wav(b"not a valid wav file header") is False


def test_transcription_service_execution(tmp_path):
    wav_path = str(tmp_path / "sample_clinical.wav")
    generate_clinical_wav(wav_path, duration_sec=1.0)
    
    with open(wav_path, "rb") as f:
        audio_bytes = f.read()

    transcript = TranscriptionService.transcribe_audio(audio_bytes, filename="sample_clinical.wav")
    assert isinstance(transcript, str)
    assert len(transcript) > 0
    assert "pain" in transcript.lower() or "clinician" in transcript.lower()
