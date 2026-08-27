import io
import pytest
from fastapi.testclient import TestClient
from app.main import app
from tests.generate_sample_audio import generate_clinical_wav

client = TestClient(app)


def test_health_endpoints():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

    root_res = client.get("/")
    assert root_res.status_code == 200
    assert root_res.json()["status"] == "healthy"


def test_ep1_parse_audio_endpoint(tmp_path):
    wav_path = str(tmp_path / "test_session.wav")
    generate_clinical_wav(wav_path, duration_sec=1.5)

    with open(wav_path, "rb") as f:
        response = client.post(
            "/assessments/parse",
            files={"file": ("test_session.wav", f, "audio/wav")}
        )

    assert response.status_code == 200
    data = response.json()
    assert "clinicalDetails" in data
    assert "subjectiveAssessments" in data
    assert "objectiveAssessment" in data
    assert "subjectiveGoals" in data
    assert "objectiveGoals" in data
    assert "recommendation" in data
    assert "patientAdvice" in data


def test_ep1_parse_invalid_file():
    response = client.post(
        "/assessments/parse",
        files={"file": ("invalid.txt", b"plain text", "text/plain")}
    )
    assert response.status_code == 400


def test_ep2_save_and_ep3_retrieve():
    payload = {
        "clinicalDetails": {
            "clinicalHistory": "Patient slipped and fell.",
            "chiefComplaint": "Knee pain",
            "duration": "1 week",
        },
        "subjectiveAssessments": [
            {
                "testName": "Knee Evaluation",
                "conclusion": "Patellar tendinitis"
            }
        ],
        "objectiveAssessment": {
            "tests": [
                {
                    "testName": "Knee Extension",
                    "unitName": "degrees",
                    "value": "0 degrees",
                    "left": "0",
                    "right": "0",
                    "comments": "Normal",
                }
            ]
        },
        "subjectiveGoals": [
            {
                "goalDetails": "Run 5k without pain",
                "targetDate": "8 weeks",
            }
        ],
        "objectiveGoals": [
            {
                "goalName": "Quad strength",
                "goalCategory": "Strength",
                "unitName": "lbs",
                "value": "50 lbs",
                "targetDate": "8 weeks",
            }
        ],
        "recommendation": [
            {
                "sessionType": "Physical Therapy",
                "sessionFrequency": "3 times per week",
            }
        ],
        "patientAdvice": {
            "adviceDetails": "Elevate and apply cold compression.",
        },
    }

    # EP2: Save
    post_res = client.post("/assessments", json=payload)
    assert post_res.status_code == 201
    post_data = post_res.json()
    assert "id" in post_data
    assessment_id = post_data["id"]

    # EP3: Retrieve by ID
    get_res = client.get(f"/assessments/{assessment_id}")
    assert get_res.status_code == 200
    get_data = get_res.json()
    assert get_data["id"] == assessment_id
    assert get_data["assessment"]["clinicalDetails"]["chiefComplaint"] == "Knee pain"


def test_ep3_retrieve_not_found():
    response = client.get("/assessments/nonexistent_id_12345")
    assert response.status_code == 404


def test_ep4_list_assessments():
    response = client.get("/assessments")
    assert response.status_code == 200
    data = response.json()
    assert "total" in data
    assert "items" in data
    assert isinstance(data["items"], list)
