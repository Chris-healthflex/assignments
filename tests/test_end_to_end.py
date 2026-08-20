import io
import pytest
from fastapi.testclient import TestClient
from app.schemas.first_assessment import FirstAssessment


def test_full_pipeline_end_to_end(test_client: TestClient, sample_wav_content: bytes):
    """
    Complete End-to-End Pipeline Verification:
    1. POST /assessments/parse with WAV file -> FirstAssessment JSON
    2. POST /assessments with JSON -> MongoDB Save with ID
    3. GET /assessments/{id} -> Retrieve exact saved document
    4. GET /assessments -> List and confirm document in list
    """
    # 1. Parse WAV
    files = {
        "file": ("clinical_session_e2e.wav", io.BytesIO(sample_wav_content), "audio/wav")
    }
    parse_resp = test_client.post("/assessments/parse", files=files)
    assert parse_resp.status_code == 200, f"Parse failed: {parse_resp.text}"

    parsed_json = parse_resp.json()
    first_assessment = FirstAssessment.model_validate(parsed_json)
    assert "knee pain" in first_assessment.clinicalDetails.chiefComplaint.lower()

    # 2. Persist to MongoDB
    save_resp = test_client.post("/assessments", json=parsed_json)
    assert save_resp.status_code == 201
    save_data = save_resp.json()
    doc_id = save_data["id"]
    assert doc_id is not None

    # 3. Retrieve from MongoDB
    get_resp = test_client.get(f"/assessments/{doc_id}")
    assert get_resp.status_code == 200
    retrieved_doc = get_resp.json()
    assert retrieved_doc["id"] == doc_id
    assert "knee pain" in retrieved_doc["assessment"]["clinicalDetails"]["chiefComplaint"].lower()

    # 4. Verify in List
    list_resp = test_client.get("/assessments")
    assert list_resp.status_code == 200
    list_data = list_resp.json()
    assert list_data["total"] >= 1
    found = any(a["id"] == doc_id for a in list_data["assessments"])
    assert found is True
