"""Integration tests for FastAPI REST API endpoints (EP1 - EP4)."""

import io
from unittest.mock import MagicMock
import pytest
from fastapi.testclient import TestClient

from app.api.deps import (
    get_assessment_repo,
    get_extraction_agent,
    get_transcriber,
)
from app.db.mongo import MongoDBManager
from app.main import app
from app.repositories.assessment_repo import AssessmentRepository
from app.schemas.assessment import (
    ClinicalDetails,
    FirstAssessment,
    ObjectiveAssessment,
    ObjectiveTest,
    PatientAdvice,
    Recommendation,
    SubjectiveAssessment,
)
from app.services.langgraph_agent import ExtractionState
from app.services.transcriber import AudioValidationError, TranscriptionError


@pytest.fixture
def test_db_manager() -> MongoDBManager:
    """Provide isolated MongoDB test manager."""
    return MongoDBManager(db_name="test_clinical_api_db", collection_name="test_api_assessments")


@pytest.fixture
def test_repo(test_db_manager: MongoDBManager) -> AssessmentRepository:
    """Provide isolated repository for API testing."""
    return AssessmentRepository(manager=test_db_manager)


@pytest.fixture
def sample_assessment_fixture() -> FirstAssessment:
    """Fixture with valid FirstAssessment instance."""
    return FirstAssessment(
        clinicalDetails=ClinicalDetails(
            clinicalHistory="Road traffic accident, left tibial condyle fracture, ORIF.",
            chiefComplaint="Left knee pain and functional walking difficulty.",
            duration={"text": "8 months"},
        ),
        subjectiveAssessments=[
            SubjectiveAssessment(testName="Pain Scale", conclusion=["Moderate pain"])
        ],
        objectiveAssessment=ObjectiveAssessment(
            tests=[
                ObjectiveTest(
                    testName="Knee Flexion",
                    unitName="degrees",
                    left="124",
                    right="130",
                    comments=["Painful at end range"],
                )
            ]
        ),
        recommendation=[
            Recommendation(sessionType="Physiotherapy", sessionFrequency="Once weekly for 4 sessions")
        ],
        patientAdvice=PatientAdvice(
            adviceDetails="Avoid prolonged standing."
        ),
    )


@pytest.fixture
def mock_transcriber() -> MagicMock:
    """Mock Whisper transcriber."""
    transcriber = MagicMock()
    transcriber.transcribe.return_value = "Patient has left knee pain and flexion left 124 right 130 degrees."
    return transcriber


@pytest.fixture
def mock_agent(sample_assessment_fixture: FirstAssessment) -> MagicMock:
    """Mock LangGraph extraction agent returning valid extraction state."""
    agent = MagicMock()
    agent.extract.return_value = {
        "transcript": "dummy transcript",
        "raw_assessment": sample_assessment_fixture,
        "evidence": {"objectiveAssessment.tests[0].left": "124"},
        "uncertain_fields": [],
        "validation_errors": [],
        "final_assessment": sample_assessment_fixture,
        "is_valid": True,
    }
    return agent


@pytest.fixture
def client(test_repo: AssessmentRepository, mock_transcriber: MagicMock, mock_agent: MagicMock) -> TestClient:
    """FastAPI TestClient with injected test repository and mocked AI services."""
    app.dependency_overrides[get_assessment_repo] = lambda: test_repo
    app.dependency_overrides[get_transcriber] = lambda: mock_transcriber
    app.dependency_overrides[get_extraction_agent] = lambda: mock_agent

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()
    test_repo.manager.close()


# ---------------------------------------------------------------------------
# EP1: POST /assessments/parse Tests
# ---------------------------------------------------------------------------

def test_ep1_parse_wav_success(client: TestClient, sample_assessment_fixture: FirstAssessment):
    """EP1: Upload valid WAV file and receive FirstAssessment JSON."""
    wav_bytes = b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x44\xac\x00\x00data\x00\x00\x00\x00"
    files = {"file": ("clinical_assessment.wav", io.BytesIO(wav_bytes), "audio/wav")}

    response = client.post("/assessments/parse", files=files)
    assert response.status_code == 200

    data = response.json()
    assert data["clinicalDetails"]["chiefComplaint"] == sample_assessment_fixture.clinicalDetails.chiefComplaint
    assert data["objectiveAssessment"]["tests"][0]["left"] == "124"

    # Verify no metadata/leakage
    assert "evidence" not in data
    assert "uncertain_fields" not in data
    assert "id" not in data


def test_ep1_parse_empty_file_rejected(client: TestClient):
    """EP1: Uploading empty 0-byte file returns 400."""
    files = {"file": ("empty.wav", io.BytesIO(b""), "audio/wav")}
    response = client.post("/assessments/parse", files=files)
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_ep1_parse_unsupported_format_rejected(client: TestClient):
    """EP1: Uploading non-audio file returns 400."""
    files = {"file": ("notes.pdf", io.BytesIO(b"dummy pdf content"), "application/pdf")}
    response = client.post("/assessments/parse", files=files)
    assert response.status_code == 400
    assert "unsupported file format" in response.json()["detail"].lower()


def test_ep1_transcription_failure_returns_500(client: TestClient, mock_transcriber: MagicMock):
    """EP1: Whisper transcription failure returns 500."""
    mock_transcriber.transcribe.side_effect = TranscriptionError("API timeout")

    wav_bytes = b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x44\xac\x00\x00data\x00\x00\x00\x00"
    files = {"file": ("session.wav", io.BytesIO(wav_bytes), "audio/wav")}

    response = client.post("/assessments/parse", files=files)
    assert response.status_code == 500
    assert "transcription failed" in response.json()["detail"].lower()


def test_ep1_low_confidence_grounding_failure_returns_422(client: TestClient, mock_agent: MagicMock):
    """EP1: Low-confidence/ungrounded extraction returns HTTP 422 with field-level details."""
    mock_agent.extract.return_value = {
        "transcript": "dummy transcript",
        "raw_assessment": FirstAssessment(),
        "evidence": {},
        "uncertain_fields": [
            {
                "field": "objectiveAssessment.tests[0].left",
                "value": "999",
                "reason": "Measurement value '999' is not found or supported in the transcript text.",
            }
        ],
        "validation_errors": [],
        "final_assessment": None,
        "is_valid": False,
    }

    wav_bytes = b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x44\xac\x00\x00data\x00\x00\x00\x00"
    files = {"file": ("session.wav", io.BytesIO(wav_bytes), "audio/wav")}

    response = client.post("/assessments/parse", files=files)
    assert response.status_code == 422

    detail = response.json()["detail"]
    assert isinstance(detail, list)
    assert any(item["field"] == "objectiveAssessment.tests[0].left" for item in detail)
    assert any("not found or supported" in item["message"] for item in detail)


# ---------------------------------------------------------------------------
# EP2: POST /assessments Tests
# ---------------------------------------------------------------------------

def test_ep2_save_assessment_success(client: TestClient, sample_assessment_fixture: FirstAssessment):
    """EP2: Save FirstAssessment JSON payload to MongoDB."""
    payload = sample_assessment_fixture.model_dump()
    response = client.post("/assessments", json=payload)

    assert response.status_code == 201
    data = response.json()
    assert "id" in data
    assert len(data["id"]) == 24
    assert data["message"] == "Assessment saved successfully"
    assert data["assessment"]["clinicalDetails"]["chiefComplaint"] == sample_assessment_fixture.clinicalDetails.chiefComplaint


def test_ep2_save_invalid_extra_field_rejected(client: TestClient, sample_assessment_fixture: FirstAssessment):
    """EP2: Reject payload with unapproved extra fields (Pydantic extra='forbid')."""
    payload = sample_assessment_fixture.model_dump()
    payload["unexpectedField"] = "should fail"

    response = client.post("/assessments", json=payload)
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# EP3: GET /assessments/{id} Tests
# ---------------------------------------------------------------------------

def test_ep3_get_assessment_by_id_success(client: TestClient, sample_assessment_fixture: FirstAssessment):
    """EP3: Retrieve existing assessment by ID returning pure FirstAssessment."""
    # First save via EP2
    save_res = client.post("/assessments", json=sample_assessment_fixture.model_dump())
    doc_id = save_res.json()["id"]

    # Now retrieve via EP3
    get_res = client.get(f"/assessments/{doc_id}")
    assert get_res.status_code == 200

    retrieved = get_res.json()
    assert retrieved["clinicalDetails"]["chiefComplaint"] == sample_assessment_fixture.clinicalDetails.chiefComplaint
    assert retrieved["objectiveAssessment"]["tests"][0]["left"] == "124"

    # Ensure pure FirstAssessment schema (no DB metadata inside retrieved JSON)
    assert "id" not in retrieved
    assert "_id" not in retrieved
    assert "created_at" not in retrieved


def test_ep3_get_missing_id_returns_404(client: TestClient):
    """EP3: Non-existent ID returns 404."""
    response = client.get("/assessments/507f1f77bcf86cd799439011")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_ep3_get_malformed_id_returns_404(client: TestClient):
    """EP3: Malformed ID returns 404."""
    response = client.get("/assessments/invalid-hex-id")
    assert response.status_code == 404


# ---------------------------------------------------------------------------
# EP4: GET /assessments Tests
# ---------------------------------------------------------------------------

def test_ep4_list_assessments_and_date_filter(client: TestClient, sample_assessment_fixture: FirstAssessment):
    """EP4: List assessments and filter by date range."""
    # Save an assessment
    client.post("/assessments", json=sample_assessment_fixture.model_dump())

    # List all
    res = client.get("/assessments")
    assert res.status_code == 200
    data = res.json()
    assert data["total"] >= 1
    assert len(data["items"]) >= 1
    assert isinstance(data["items"], list)

    # Date filter with wide valid window
    filter_res = client.get("/assessments?start_date=2026-01-01T00:00:00Z&end_date=2026-12-31T23:59:59Z")
    assert filter_res.status_code == 200
    assert filter_res.json()["total"] >= 1


def test_ep4_invalid_date_format_rejected(client: TestClient):
    """EP4: Malformed date string returns 400 Bad Request."""
    res = client.get("/assessments?start_date=not-a-date")
    assert res.status_code == 400
    assert "invalid date" in res.json()["detail"].lower()


# ---------------------------------------------------------------------------
# Health Check Test
# ---------------------------------------------------------------------------

def test_health_check_endpoint(client: TestClient):
    """Infrastructure: Verify /health returns 200 and healthy status."""
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "healthy"
