import json
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from app.main import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200

@patch("app.api.assessments.MongoDB")
def test_create_assessment(mock_mongo):
    mock_db = MagicMock()
    mock_db.save_assessment.return_value = "12345"
    mock_mongo.return_value = mock_db
    
    payload = {
        "clinicalDetails": {
            "clinicalHistory": "History",
            "chiefComplaint": "Complaint",
            "duration": "1 week"
        },
        "subjectiveAssessments": [],
        "objectiveAssessment": {
            "tests": []
        },
        "subjectiveGoals": [],
        "objectiveGoals": [],
        "recommendation": [],
        "patientAdvice": {
            "adviceDetails": ""
        }
    }
    
    response = client.post("/assessments", json=payload)
    assert response.status_code == 201

@patch("app.api.assessments.MongoDB")
def test_get_assessment(mock_mongo):
    mock_db = MagicMock()
    mock_db.get_assessment.return_value = {"_id": "12345", "clinicalDetails": {}}
    mock_mongo.return_value = mock_db
    
    valid_id = "507f1f77bcf86cd799439011"
    response = client.get(f"/assessments/{valid_id}")
    assert response.status_code == 200

def test_get_assessment_invalid_id():
    response = client.get("/assessments/invalid_id")
    assert response.status_code == 400
    
@patch("app.api.assessments.MongoDB")
def test_get_assessment_not_found(mock_mongo):
    mock_db = MagicMock()
    mock_db.get_assessment.return_value = None
    mock_mongo.return_value = mock_db
    
    valid_id = "507f1f77bcf86cd799439011"
    response = client.get(f"/assessments/{valid_id}")
    assert response.status_code == 404

@patch("app.api.assessments.MongoDB")
def test_list_assessments(mock_mongo):
    mock_db = MagicMock()
    mock_cursor = MagicMock()
    mock_cursor.sort.return_value = [{"_id": "1", "data": "test"}]
    mock_db.collection.find.return_value = mock_cursor
    mock_mongo.return_value = mock_db
    
    response = client.get("/assessments")
    assert response.status_code == 200

def test_list_assessments_invalid_date_range():
    response = client.get("/assessments?date_from=2023-01-02&date_to=2023-01-01")
    assert response.status_code == 400

def test_parse_assessment_invalid_extension():
    files = {"file": ("test.txt", b"hello")}
    response = client.post("/assessments/parse", files=files)
    assert response.status_code == 400

def test_parse_assessment_empty_file():
    files = {"file": ("test.wav", b"")}
    response = client.post("/assessments/parse", files=files)
    assert response.status_code == 400


