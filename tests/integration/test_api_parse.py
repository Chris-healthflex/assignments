import io
import wave
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock
import pytest

def create_wav_bytes(duration_sec=1.0, sample_rate=16000):
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        n_frames = int(duration_sec * sample_rate)
        wf.writeframes(b'\x00\x00' * n_frames)
    return buf.getvalue()

def test_parse_endpoint_mocked(client):
    # Mock pipeline to return a valid FirstAssessment
    with patch("app.api.routes.parse.process_audio_file", new=AsyncMock()) as mock_proc:
        mock_proc.return_value = {
            "clinicalDetails": {"clinicalHistory": "", "chiefComplaint": "headache", "duration": "2 days"},
            "subjectiveAssessments": [],
            "objectiveAssessment": {"tests": []},
            "subjectiveGoals": [],
            "objectiveGoals": [],
            "recommendation": [],
            "patientAdvice": {"adviceDetails": ""}
        }
        response = client.post(
            "/api/v1/assessments/parse",
            files={"file": ("test.wav", create_wav_bytes(), "audio/wav")}
        )
    assert response.status_code == 200
    assert response.json()["clinicalDetails"]["chiefComplaint"] == "headache"