from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from app.main import app
from app.models.assessment import FirstAssessment
from app.services.extraction import (
    ConfidenceIssue,
    ExtractionConfidence,
)

client = TestClient(app)


def make_assessment_payload():
    return {
        "clinicalDetails": {
            "clinicalHistory": "History",
            "chiefComplaint": "Complaint",
            "duration": "1 week",
        },
        "subjectiveAssessments": [],
        "objectiveAssessment": {
            "tests": [],
        },
        "subjectiveGoals": [],
        "objectiveGoals": [],
        "recommendation": [],
        "patientAdvice": {
            "adviceDetails": "",
        },
    }


def make_first_assessment():
    return FirstAssessment(
        clinicalDetails={
            "clinicalHistory": "History",
            "chiefComplaint": "Complaint",
            "duration": "1 week",
        },
        subjectiveAssessments=[],
        objectiveAssessment={
            "tests": [],
        },
        subjectiveGoals=[],
        objectiveGoals=[],
        recommendation=[],
        patientAdvice={
            "adviceDetails": "",
        },
    )


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


@patch("app.api.assessments.MongoDB")
def test_create_assessment(mock_mongo):
    mock_db = MagicMock()
    mock_db.save_assessment.return_value = "12345"
    mock_mongo.return_value = mock_db

    response = client.post(
        "/assessments",
        json=make_assessment_payload(),
    )

    assert response.status_code == 201

    data = response.json()

    assert data["id"] == "12345"
    assert data["message"] == (
        "Assessment saved successfully"
    )

    mock_db.save_assessment.assert_called_once()


@patch("app.api.assessments.MongoDB")
def test_get_assessment(mock_mongo):
    mock_db = MagicMock()

    assessment = make_first_assessment().model_dump(
        by_alias=False
    )

    assessment["_id"] = "12345"

    mock_db.get_assessment.return_value = assessment
    mock_mongo.return_value = mock_db

    valid_id = "507f1f77bcf86cd799439011"

    response = client.get(
        f"/assessments/{valid_id}"
    )

    assert response.status_code == 200

    data = response.json()

    assert data["_id"] == "12345"
    assert data["clinicalDetails"]["clinicalHistory"] == (
        "History"
    )
    assert data["patientAdvice"]["adviceDetails"] == ""

    mock_db.get_assessment.assert_called_once_with(
        valid_id
    )


def test_get_assessment_invalid_id():
    response = client.get(
        "/assessments/invalid_id"
    )

    assert response.status_code == 400


@patch("app.api.assessments.MongoDB")
def test_get_assessment_not_found(mock_mongo):
    mock_db = MagicMock()
    mock_db.get_assessment.return_value = None
    mock_mongo.return_value = mock_db

    valid_id = "507f1f77bcf86cd799439011"

    response = client.get(
        f"/assessments/{valid_id}"
    )

    assert response.status_code == 404

    mock_db.get_assessment.assert_called_once_with(
        valid_id
    )


@patch("app.api.assessments.MongoDB")
def test_list_assessments(mock_mongo):
    mock_db = MagicMock()

    assessment = make_first_assessment().model_dump(
        by_alias=False
    )

    assessment["_id"] = "1"

    mock_cursor = MagicMock()
    mock_cursor.sort.return_value = [assessment]

    mock_db.collection.find.return_value = mock_cursor
    mock_mongo.return_value = mock_db

    response = client.get(
        "/assessments"
    )

    assert response.status_code == 200

    data = response.json()

    assert len(data) == 1
    assert data[0]["_id"] == "1"
    assert data[0]["clinicalDetails"]["clinicalHistory"] == (
        "History"
    )

    mock_db.collection.find.assert_called_once_with({})


def test_list_assessments_invalid_date_range():
    response = client.get(
        "/assessments"
        "?date_from=2023-01-02"
        "&date_to=2023-01-01"
    )

    assert response.status_code == 400


@patch("app.api.assessments.MongoDB")
def test_list_assessments_with_date_filter(
    mock_mongo,
):
    assessment = make_first_assessment().model_dump(
        by_alias=False
    )

    assessment["_id"] = "1"
    assessment["createdAt"] = (
        "2023-01-05T12:00:00+00:00"
    )

    mock_db = MagicMock()

    mock_cursor = MagicMock()
    mock_cursor.sort.return_value = [assessment]

    mock_db.collection.find.return_value = mock_cursor
    mock_mongo.return_value = mock_db

    response = client.get(
        "/assessments"
        "?date_from=2023-01-01"
        "&date_to=2023-01-10"
    )

    assert response.status_code == 200

    data = response.json()

    assert len(data) == 1
    assert data[0]["_id"] == "1"
    assert data[0]["clinicalDetails"]["clinicalHistory"] == (
        "History"
    )
    assert "createdAt" in data[0]

    mock_db.collection.find.assert_called_once()

    query = mock_db.collection.find.call_args[0][0]

    assert "createdAt" in query
    assert "$gte" in query["createdAt"]
    assert "$lte" in query["createdAt"]


def test_parse_assessment_invalid_extension():
    files = {
        "file": (
            "test.txt",
            b"hello",
            "text/plain",
        )
    }

    response = client.post(
        "/assessments/parse",
        files=files,
    )

    assert response.status_code == 400


def test_parse_assessment_empty_file():
    files = {
        "file": (
            "test.wav",
            b"",
            "audio/wav",
        )
    }

    response = client.post(
        "/assessments/parse",
        files=files,
    )

    assert response.status_code == 400


@patch("app.api.assessments.WhisperTranscriber")
@patch("app.api.assessments.build_assessment_graph")
def test_parse_assessment_success(
    mock_graph_builder,
    mock_transcriber,
):
    transcript = (
        "Patient has left knee pain for eight months."
    )

    mock_transcriber.return_value.transcribe.return_value = (
        transcript
    )

    assessment = FirstAssessment(
        clinicalDetails={
            "clinicalHistory": "History",
            "chiefComplaint": "left knee pain",
            "duration": "eight months",
        },
        subjectiveAssessments=[],
        objectiveAssessment={
            "tests": [],
        },
        subjectiveGoals=[],
        objectiveGoals=[],
        recommendation=[],
        patientAdvice={
            "adviceDetails": "",
        },
    )

    confidence = ExtractionConfidence(
        overall_confidence=0.95,
        issues=[],
    )

    mock_graph = MagicMock()

    mock_graph.invoke.return_value = {
        "assessment": assessment,
        "confidence": confidence,
    }

    mock_graph_builder.return_value = mock_graph

    files = {
        "file": (
            "test.wav",
            b"fake wav content",
            "audio/wav",
        )
    }

    response = client.post(
        "/assessments/parse",
        files=files,
    )

    assert response.status_code == 200

    data = response.json()

    assert data["clinicalDetails"]["chiefComplaint"] == (
        "left knee pain"
    )
    assert data["clinicalDetails"]["duration"] == (
        "eight months"
    )

    assert data["subjectiveAssessments"] == []
    assert data["objectiveAssessment"]["tests"] == []
    assert data["subjectiveGoals"] == []
    assert data["objectiveGoals"] == []
    assert data["recommendation"] == []
    assert data["patientAdvice"]["adviceDetails"] == ""

    mock_transcriber.return_value.transcribe.assert_called_once()

    mock_graph.invoke.assert_called_once_with(
        {
            "transcript": transcript,
        }
    )


@patch("app.api.assessments.WhisperTranscriber")
@patch("app.api.assessments.build_assessment_graph")
def test_parse_assessment_low_confidence_returns_422(
    mock_graph_builder,
    mock_transcriber,
):
    transcript = (
        "Patient has knee extension of negic 5 degrees."
    )

    mock_transcriber.return_value.transcribe.return_value = (
        transcript
    )

    assessment = FirstAssessment(
        clinicalDetails={
            "clinicalHistory": "",
            "chiefComplaint": "knee problem",
            "duration": "",
        },
        subjectiveAssessments=[],
        objectiveAssessment={
            "tests": [],
        },
        subjectiveGoals=[],
        objectiveGoals=[],
        recommendation=[],
        patientAdvice={
            "adviceDetails": "",
        },
    )

    confidence = ExtractionConfidence(
        overall_confidence=0.55,
        issues=[
            ConfidenceIssue(
                field_path=(
                    "objectiveAssessment.tests[0].right"
                ),
                confidence=0.55,
                reason=(
                    "Ambiguous transcription marker "
                    "'negic' detected."
                ),
            )
        ],
    )

    mock_graph = MagicMock()

    mock_graph.invoke.return_value = {
        "assessment": assessment,
        "confidence": confidence,
    }

    mock_graph_builder.return_value = mock_graph

    files = {
        "file": (
            "test.wav",
            b"fake wav content",
            "audio/wav",
        )
    }

    response = client.post(
        "/assessments/parse",
        files=files,
    )

    assert response.status_code == 422

    body = response.json()

    assert body["detail"]["error"] == (
        "Low-confidence clinical extraction."
    )

    assert body["detail"]["threshold"] == 0.70

    assert len(body["detail"]["issues"]) == 1

    issue = body["detail"]["issues"][0]

    assert issue["field_path"] == (
        "objectiveAssessment.tests[0].right"
    )

    assert issue["confidence"] == 0.55

    assert "negic" in issue["reason"]

    mock_transcriber.return_value.transcribe.assert_called_once()

    mock_graph.invoke.assert_called_once_with(
        {
            "transcript": transcript,
        }
    )