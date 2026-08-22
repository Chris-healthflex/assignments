import io
import datetime
import json
from unittest.mock import MagicMock, patch
from bson import ObjectId
from app.models.assessment import FirstAssessment
from app.services.extraction import fallback_validate

# Helper to create a fake WAV bytes structure
def make_mock_wav(content: bytes = b"RIFFxxxxWAVEfmt ") -> bytes:
    # Needs to be at least 12 bytes and start with RIFF...WAVE
    return content

def test_health_check(client, mock_mongo_client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_parse_invalid_extension(client):
    # Not ending in .wav
    response = client.post(
        "/assessments/parse",
        files={"file": ("test.mp3", b"fake audio content", "audio/mpeg")}
    )
    assert response.status_code == 400
    assert "Only WAV files are allowed" in response.json()["detail"]

def test_parse_empty_file(client):
    response = client.post(
        "/assessments/parse",
        files={"file": ("test.wav", b"", "audio/wav")}
    )
    assert response.status_code == 400
    assert "audio file is empty" in response.json()["detail"].lower()

def test_parse_bad_wav_header(client):
    # Ends in .wav but lacks RIFF WAVE header
    response = client.post(
        "/assessments/parse",
        files={"file": ("test.wav", b"too_short", "audio/wav")}
    )
    assert response.status_code == 400
    assert "not a valid WAV" in response.json()["detail"]

@patch("app.api.routes.assessments.transcribe_audio")
@patch("app.api.routes.assessments.app_graph.invoke")
@patch("app.api.routes.assessments.create_assessment")
def test_parse_success(
    mock_create,
    mock_invoke,
    mock_transcribe,
    client,
    mock_mongo_client
):
    mock_transcribe.return_value = "Patient has shoulder pain for two weeks."

    
    mock_assessment_dict = {
        "clinicalDetails": {
            "clinicalHistory": "Patient reports pain.",
            "chiefComplaint": "Shoulder pain",
            "duration": "2 weeks"
        },
        "subjectiveAssessments": [],
        "objectiveAssessment": {"tests": []},
        "subjectiveGoals": [],
        "objectiveGoals": [],
        "recommendation": [],
        "patientAdvice": {"adviceDetails": "Rest"}
    }
    
    mock_invoke.return_value = {
        "transcript": "Patient has shoulder pain for two weeks.",
        "validation_errors": [],
        "first_assessment": mock_assessment_dict
    }
    
    mock_create.return_value = "60c72b2f9b1d8b2d88888888"

    wav_data = make_mock_wav()
    response = client.post(
        "/assessments/parse",
        files={"file": ("test.wav", wav_data, "audio/wav")}
    )
    
    assert response.status_code == 201
    res_json = response.json()
    assert res_json["id"] == "60c72b2f9b1d8b2d88888888"
    assert res_json["assessment"]["clinicalDetails"]["chiefComplaint"] == "Shoulder pain"
    assert res_json["assessment"]["clinicalDetails"]["duration"] == "2 weeks"

@patch("app.api.routes.assessments.transcribe_audio")
@patch("app.api.routes.assessments.app_graph.invoke")
def test_parse_low_confidence(
    mock_invoke,
    mock_transcribe,
    client,
    mock_mongo_client
):
    mock_transcribe.return_value = "Patient has shoulder pain."
    
    # Simulate a validation failure due to low confidence
    mock_errors = [
        {
            "field": "clinicalDetails.duration",
            "reason": "Not mentioned in the transcript but extracted as 2 weeks",
            "confidence": 0.35
        }
    ]
    
    mock_invoke.return_value = {
        "transcript": "Patient has shoulder pain.",
        "validation_errors": mock_errors,
        "first_assessment": None
    }

    wav_data = make_mock_wav()
    response = client.post(
        "/assessments/parse",
        files={"file": ("test.wav", wav_data, "audio/wav")}
    )
    
    assert response.status_code == 422
    res_json = response.json()
    assert len(res_json["detail"]) == 1
    assert res_json["detail"][0]["field"] == "clinicalDetails.duration"
    assert res_json["detail"][0]["confidence"] == 0.35

@patch("app.database.mongodb.get_collection")
def test_get_assessment_success(mock_get_collection, client):
    oid = ObjectId("60c72b2f9b1d8b2d88888888")
    mock_doc = {
        "_id": oid,
        "created_at": datetime.datetime.now(datetime.timezone.utc),
        "updated_at": datetime.datetime.now(datetime.timezone.utc),
        "clinicalDetails": {
            "clinicalHistory": "History details",
            "chiefComplaint": "Back pain",
            "duration": "1 month"
        },
        "subjectiveAssessments": [],
        "objectiveAssessment": {"tests": []},
        "subjectiveGoals": [],
        "objectiveGoals": [],
        "recommendation": [],
        "patientAdvice": {"adviceDetails": "Keep active"}
    }
    
    mock_col = MagicMock()
    mock_col.find_one.return_value = mock_doc
    mock_get_collection.return_value = mock_col

    response = client.get("/assessments/60c72b2f9b1d8b2d88888888")
    assert response.status_code == 200
    res_json = response.json()
    assert res_json["id"] == "60c72b2f9b1d8b2d88888888"
    assert res_json["assessment"]["clinicalDetails"]["chiefComplaint"] == "Back pain"
    # Ensure extra db fields were not leaked inside assessment body
    assert "created_at" not in res_json["assessment"]

def test_get_assessment_invalid_id(client):
    response = client.get("/assessments/invalid-id")
    assert response.status_code == 400
    assert "Invalid database ID format" in response.json()["detail"]

@patch("app.database.mongodb.get_collection")
def test_get_assessment_not_found(mock_get_collection, client):
    mock_col = MagicMock()
    mock_col.find_one.return_value = None
    mock_get_collection.return_value = mock_col

    response = client.get("/assessments/60c72b2f9b1d8b2d88888889")
    assert response.status_code == 404
    assert "Clinical assessment not found" in response.json()["detail"]

@patch("app.database.mongodb.get_collection")
def test_list_assessments_success(mock_get_collection, client):
    oid = ObjectId("60c72b2f9b1d8b2d88888888")
    mock_docs = [
        {
            "_id": oid,
            "created_at": datetime.datetime.now(datetime.timezone.utc),
            "clinicalDetails": {
                "clinicalHistory": "Details",
                "chiefComplaint": "Knee pain",
                "duration": "1 week"
            },
            "subjectiveAssessments": [],
            "objectiveAssessment": {"tests": []},
            "subjectiveGoals": [],
            "objectiveGoals": [],
            "recommendation": [],
            "patientAdvice": {"adviceDetails": "Ice pack"}
        }
    ]
    
    mock_col = MagicMock()
    mock_cursor = MagicMock()
    mock_cursor.sort.return_value.skip.return_value.limit.return_value = mock_docs
    mock_col.find.return_value = mock_cursor
    mock_get_collection.return_value = mock_col

    response = client.get("/assessments?limit=10&offset=0")
    assert response.status_code == 200
    res_json = response.json()
    assert isinstance(res_json, list)
    assert len(res_json) == 1
    assert res_json[0]["id"] == "60c72b2f9b1d8b2d88888888"
    assert res_json[0]["assessment"]["clinicalDetails"]["chiefComplaint"] == "Knee pain"

def test_fallback_validate_logic():
    # Test that fallback_validate handles invalid fields gracefully by using defaults and tagging with confidence=0.0
    bad_data = {
        "clinicalDetails": {
            # chiefComplaint is missing, duration is invalid type
            "clinicalHistory": "Yes",
            "duration": 123  # should be str
        },
        "subjectiveAssessments": "Not a list!"  # should be List
    }
    
    valid_dict, errors = fallback_validate(bad_data)
    
    # clinicalDetails should fall back to default or cast, depending on pydantic parsing
    # Here, 'duration' = 123 is coerced to "123" by default Pydantic string validation, which is valid!
    # But subjectiveAssessments should be an empty list, and marked in errors.
    assert isinstance(valid_dict["subjectiveAssessments"], list)
    assert len(valid_dict["subjectiveAssessments"]) == 0
    assert "subjectiveAssessments" in errors
    assert errors["subjectiveAssessments"]["confidence"] == 0.0
