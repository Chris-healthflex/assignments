from datetime import datetime
from unittest.mock import Mock

import mongomock
from fastapi.testclient import TestClient

from app.api import assessments as assessment_routes
from app.main import app
from app.models.first_assessment import FirstAssessment


client = TestClient(app)


def sample_assessment() -> dict:
    return FirstAssessment().model_dump()


def test_parse_success_with_transcription_and_extraction_mocked(monkeypatch) -> None:
    monkeypatch.setattr(assessment_routes.WhisperTranscriber, "transcribe", lambda self, path: "transcript")
    monkeypatch.setattr(assessment_routes, "_parse_transcript", lambda transcript: FirstAssessment())
    response = client.post("/assessments/parse", files={"file": ("session.wav", b"RIFF audio", "audio/wav")})
    assert response.status_code == 200
    assert response.json() == sample_assessment()


def test_parse_rejects_non_wav_upload() -> None:
    response = client.post("/assessments/parse", files={"file": ("session.mp3", b"audio", "audio/mpeg")})
    assert response.status_code == 422


def test_parse_returns_422_for_forced_low_confidence(monkeypatch) -> None:
    class LowConfidenceGraph:
        def __init__(self, model_name, api_key):
            pass

        def extract(self, transcript):
            return {
                "assessment": FirstAssessment(),
                "confidence": {"clinicalDetails.chiefComplaint": 0.3},
            }

    monkeypatch.setattr(assessment_routes.WhisperTranscriber, "transcribe", lambda self, path: "transcript")
    monkeypatch.setattr(assessment_routes, "ClinicalExtractionGraph", LowConfidenceGraph)
    response = client.post("/assessments/parse", files={"file": ("session.wav", b"RIFF audio", "audio/wav")})
    assert response.status_code == 422
    assert response.json()["detail"]["fields"][0]["field"] == "clinicalDetails.chiefComplaint"


def test_parse_rejects_grounding_failure_even_with_high_model_confidence(monkeypatch) -> None:
    class GroundingFailureGraph:
        def __init__(self, model_name, api_key):
            pass

        def extract(self, transcript):
            assessment = FirstAssessment()
            assessment.clinicalDetails.duration = "140 years"
            return {
                "assessment": assessment,
                "confidence": {"clinicalDetails.duration": 0.99},
            }

    monkeypatch.setattr(assessment_routes.WhisperTranscriber, "transcribe", lambda self, path: "The patient has knee pain.")
    monkeypatch.setattr(assessment_routes, "ClinicalExtractionGraph", GroundingFailureGraph)
    response = client.post("/assessments/parse", files={"file": ("session.wav", b"RIFF audio", "audio/wav")})
    assert response.status_code == 422
    assert response.json()["detail"]["fields"][0]["field"] == "clinicalDetails.duration"


def test_create_success_with_database_mocked(monkeypatch) -> None:
    expected = Mock(id="abc", created_at=datetime(2026, 8, 21), **sample_assessment())
    monkeypatch.setattr(assessment_routes, "save_assessment", lambda assessment: expected)
    response = client.post("/assessments", json=sample_assessment())
    assert response.status_code == 201
    assert response.json()["id"] == "abc"


def test_create_rejects_unknown_schema_field() -> None:
    payload = sample_assessment()
    payload["unknown"] = "not allowed"
    response = client.post("/assessments", json=payload)
    assert response.status_code == 422


def test_get_by_id_success_with_database_mocked(monkeypatch) -> None:
    expected = Mock(id="abc", created_at=datetime(2026, 8, 21), **sample_assessment())
    monkeypatch.setattr(assessment_routes, "get_assessment", lambda assessment_id: expected)
    response = client.get("/assessments/abc")
    assert response.status_code == 200
    assert response.json()["id"] == "abc"


def test_get_by_id_returns_404(monkeypatch) -> None:
    monkeypatch.setattr(assessment_routes, "get_assessment", lambda assessment_id: None)
    response = client.get("/assessments/missing")
    assert response.status_code == 404


def test_list_success_with_database_mocked(monkeypatch) -> None:
    monkeypatch.setattr(assessment_routes, "list_assessments", lambda from_date, to_date: [])
    response = client.get("/assessments?from_date=2026-08-01T00:00:00&to_date=2026-08-31T23:59:59")
    assert response.status_code == 200
    assert response.json() == []


def test_list_rejects_malformed_date() -> None:
    response = client.get("/assessments?from_date=not-a-date")
    assert response.status_code == 400


def test_mongomock_round_trip(monkeypatch) -> None:
    collection = mongomock.MongoClient()["test"]["assessments"]
    monkeypatch.setattr("app.db.assessments.get_collection", lambda: collection)
    from app.db.assessments import get_assessment, list_assessments, save_assessment

    saved = save_assessment(FirstAssessment())
    assert get_assessment(saved.id).id == saved.id
    assert len(list_assessments()) == 1
