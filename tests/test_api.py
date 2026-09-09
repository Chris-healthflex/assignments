from __future__ import annotations

import io
import sys
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.main import app
from app.schemas import FirstAssessment

client = TestClient(app)


def test_health_endpoint():
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"


def test_parse_rejects_non_wav_file():
    files = {"file": ("test.mp3", b"dummy audio content", "audio/mpeg")}
    res = client.post("/assessments/parse", files=files)
    assert res.status_code == 400
    assert "must be a .wav" in res.json()["detail"]


def test_parse_rejects_empty_wav():
    files = {"file": ("empty.wav", b"", "audio/wav")}
    res = client.post("/assessments/parse", files=files)
    assert res.status_code == 400


def test_crud_assessment_lifecycle():
    sample_assessment = FirstAssessment.model_validate({
        "clinicalDetails": {
            "clinicalHistory": "Past knee arthroscopy",
            "chiefComplaint": "Knee pain on exertion",
            "duration": "4 weeks",
        },
        "subjectiveAssessments": [
            {"testName": "VAS", "conclusion": "6/10"}
        ],
        "objectiveAssessment": {
            "tests": [
                {"testName": "Knee Flexion", "unitName": "degrees", "value": "120", "left": "120", "right": "135", "comments": ""}
            ]
        },
        "recommendation": [
            {"sessionType": "Physio", "sessionFrequency": "2x weekly"}
        ]
    })

    # 1. Create Assessment (EP2)
    create_res = client.post(
        "/assessments",
        json={"assessment": sample_assessment.model_dump(), "meta": {"test": True}}
    )
    assert create_res.status_code == 201
    created_data = create_res.json()
    assert "id" in created_data
    assert created_data["assessment"]["clinicalDetails"]["chiefComplaint"] == "Knee pain on exertion"
    doc_id = created_data["id"]

    # 2. Get Assessment by ID (EP3)
    get_res = client.get(f"/assessments/{doc_id}")
    assert get_res.status_code == 200
    fetched_data = get_res.json()
    assert fetched_data["id"] == doc_id
    assert fetched_data["assessment"]["clinicalDetails"]["duration"] == "4 weeks"

    # 3. List Assessments (EP4)
    list_res = client.get("/assessments?limit=10")
    assert list_res.status_code == 200
    items = list_res.json()
    assert isinstance(items, list)
    assert any(item["id"] == doc_id for item in items)


def test_get_nonexistent_assessment():
    res = client.get("/assessments/507f1f77bcf86cd799439011")
    assert res.status_code == 404


def test_parse_valid_wav_audio():
    wav_path = Path("clinical_assessment.wav")
    if wav_path.exists():
        with open(wav_path, "rb") as f:
            files = {"file": ("clinical_assessment.wav", f, "audio/wav")}
            res = client.post("/assessments/parse", files=files, data={"save": "true"})
        assert res.status_code == 200
        data = res.json()
        assert "clinicalDetails" in data
        assert data["clinicalDetails"]["chiefComplaint"] != ""
        assert "X-Extraction-Confidence" in res.headers
        assert float(res.headers["X-Extraction-Confidence"]) >= 0.70

