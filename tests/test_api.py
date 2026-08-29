from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.schemas.assessment import (
    ClinicalDetails,
    FirstAssessment,
    ObjectiveAssessment,
    PatientAdvice,
)


def _sample_assessment() -> FirstAssessment:
    return FirstAssessment(
        clinicalDetails=ClinicalDetails(
            clinicalHistory="", chiefComplaint="Knee pain", duration="3 weeks"
        ),
        subjectiveAssessments=[],
        objectiveAssessment=ObjectiveAssessment(tests=[]),
        subjectiveGoals=[],
        objectiveGoals=[],
        recommendation=[],
        patientAdvice=PatientAdvice(adviceDetails=""),
    )


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@patch("app.api.assessments.parse_wav_to_assessment")
def test_parse_endpoint_returns_first_assessment(mock_parse, client):
    mock_parse.return_value = _sample_assessment()
    files = {"file": ("clinical_assessment.wav", b"fake-bytes", "audio/wav")}
    response = client.post("/assessments/parse", files=files)
    assert response.status_code == 200
    body = response.json()
    assert body["clinicalDetails"]["chiefComplaint"] == "Knee pain"
    assert isinstance(body["subjectiveAssessments"], list)


def test_parse_endpoint_rejects_non_wav(client):
    files = {"file": ("notes.txt", b"hello", "text/plain")}
    response = client.post("/assessments/parse", files=files)
    assert response.status_code == 400


@patch("app.api.assessments.parse_wav_to_assessment")
def test_parse_endpoint_returns_422_on_low_confidence(mock_parse, client):
    from app.services.assessment_service import ConfidenceTooLowError

    mock_parse.side_effect = ConfidenceTooLowError(
        [{"field": "clinicalDetails.duration", "confidence": 0.2, "reason": "not stated"}]
    )
    files = {"file": ("clinical_assessment.wav", b"fake-bytes", "audio/wav")}
    response = client.post("/assessments/parse", files=files)
    assert response.status_code == 422
    assert response.json()["detail"]["fields"][0]["field"] == "clinicalDetails.duration"


@patch("app.api.assessments.save_assessment", new_callable=AsyncMock)
def test_create_assessment_endpoint(mock_save, client):
    mock_save.return_value = "507f1f77bcf86cd799439011"
    payload = _sample_assessment().model_dump()
    response = client.post("/assessments", json=payload)
    assert response.status_code == 201
    assert response.json() == {"id": "507f1f77bcf86cd799439011"}


@patch("app.api.assessments.get_assessment", new_callable=AsyncMock)
def test_get_assessment_not_found(mock_get, client):
    mock_get.return_value = None
    response = client.get("/assessments/000000000000000000000000")
    assert response.status_code == 404


@patch("app.api.assessments.get_assessment", new_callable=AsyncMock)
def test_get_assessment_found(mock_get, client):
    mock_get.return_value = {"id": "abc123", **_sample_assessment().model_dump()}
    response = client.get("/assessments/abc123")
    assert response.status_code == 200
    assert response.json()["id"] == "abc123"


@patch("app.api.assessments.list_assessments", new_callable=AsyncMock)
def test_list_assessments(mock_list, client):
    mock_list.return_value = [{"id": "abc123", **_sample_assessment().model_dump()}]
    response = client.get("/assessments")
    assert response.status_code == 200
    assert len(response.json()) == 1
