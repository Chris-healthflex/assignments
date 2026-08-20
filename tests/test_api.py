import io
from datetime import datetime, timezone
import pytest
from fastapi.testclient import TestClient
from app.schemas.first_assessment import FirstAssessment


def test_health_check(test_client: TestClient):
    """Verify health check endpoint."""
    response = test_client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data


def test_parse_assessment_endpoint_success(test_client: TestClient, sample_wav_content: bytes):
    """Verify POST /assessments/parse with valid WAV upload."""
    files = {
        "file": ("session_recording.wav", io.BytesIO(sample_wav_content), "audio/wav")
    }
    response = test_client.post("/assessments/parse", files=files)
    assert response.status_code == 200

    data = response.json()
    # Check structure
    assert "clinicalDetails" in data
    assert "subjectiveAssessments" in data
    assert "objectiveAssessment" in data
    assert "subjectiveGoals" in data
    assert "objectiveGoals" in data
    assert "recommendation" in data
    assert "patientAdvice" in data

    # Verify chiefComplaint was extracted
    assert "knee pain" in data["clinicalDetails"]["chiefComplaint"].lower()
    assert "3 weeks" in data["clinicalDetails"]["duration"].lower()



def test_parse_assessment_invalid_extension(test_client: TestClient):
    """Verify POST /assessments/parse rejects non-wav files."""
    files = {
        "file": ("session.mp3", io.BytesIO(b"fake mp3 audio"), "audio/mpeg")
    }
    response = test_client.post("/assessments/parse", files=files)
    assert response.status_code == 400
    assert "Unsupported file format" in response.json()["detail"]


def test_parse_assessment_empty_file(test_client: TestClient):
    """Verify POST /assessments/parse rejects empty file."""
    files = {
        "file": ("empty.wav", io.BytesIO(b""), "audio/wav")
    }
    response = test_client.post("/assessments/parse", files=files)
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_parse_assessment_corrupted_wav(test_client: TestClient):
    """Verify POST /assessments/parse rejects corrupted WAV headers."""
    corrupted_bytes = b"RIFF\x00\x00\x00\x00WAVE" + b"\x00" * 40
    files = {
        "file": ("corrupt.wav", io.BytesIO(corrupted_bytes), "audio/wav")
    }
    response = test_client.post("/assessments/parse", files=files)
    assert response.status_code == 400
    assert "invalid or corrupted" in response.json()["detail"].lower() or "failed to parse" in response.json()["detail"].lower()


def test_save_and_get_assessment(test_client: TestClient, sample_first_assessment: FirstAssessment):
    """Verify saving an assessment and retrieving it by ID."""
    payload = sample_first_assessment.model_dump()

    # 1. Save assessment
    save_resp = test_client.post("/assessments", json=payload)
    assert save_resp.status_code == 201
    save_data = save_resp.json()
    assert "id" in save_data
    assert save_data["message"] == "Assessment saved successfully"
    assert "created_at" in save_data
    assessment_id = save_data["id"]

    # 2. Get assessment by ID
    get_resp = test_client.get(f"/assessments/{assessment_id}")
    assert get_resp.status_code == 200
    get_data = get_resp.json()
    assert get_data["id"] == assessment_id
    assert get_data["assessment"]["clinicalDetails"]["chiefComplaint"] == "Knee pain after running"


def test_get_assessment_not_found(test_client: TestClient):
    """Verify GET /assessments/{id} returns 404 for non-existent ID."""
    response = test_client.get("/assessments/non-existent-id-12345")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_list_assessments_with_date_filter(test_client: TestClient, sample_first_assessment: FirstAssessment):
    """Verify GET /assessments listing and date filtering."""
    payload = sample_first_assessment.model_dump()
    test_client.post("/assessments", json=payload)
    test_client.post("/assessments", json=payload)

    # List all
    list_resp = test_client.get("/assessments")
    assert list_resp.status_code == 200
    list_data = list_resp.json()
    assert list_data["total"] >= 2

    # List with today's date filter
    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    filtered_resp = test_client.get(f"/assessments?date={today_str}")
    assert filtered_resp.status_code == 200
    filtered_data = filtered_resp.json()
    assert filtered_data["total"] >= 2

    # List with non-matching date filter
    empty_resp = test_client.get("/assessments?date=1999-01-01")
    assert empty_resp.status_code == 200
    assert empty_resp.json()["total"] == 0



