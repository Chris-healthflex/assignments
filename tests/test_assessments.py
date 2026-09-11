from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app
from app.models.assessment import FirstAssessment
from app.models.confidence import ConfidenceReport


client = TestClient(app)


def make_assessment() -> FirstAssessment:
    return FirstAssessment(
        clinicalDetails={
            "clinicalHistory": "Test history",
            "chiefComplaint": "Test complaint",
            "duration": "1 day",
        },
        subjectiveAssessments=[],
        objectiveAssessment={"tests": []},
        subjectiveGoals=[],
        objectiveGoals=[],
        recommendation=[],
        patientAdvice={"adviceDetails": ""},
    )


def test_health_check():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_confidence_below_threshold_returns_422():
    low_confidence = ConfidenceReport(
        clinicalDetails=0.95,
        subjectiveAssessments=0.95,
        objectiveAssessment=0.95,
        subjectiveGoals=0.95,
        objectiveGoals=0.95,
        recommendation=0.95,
        patientAdvice=0.50,
    )

    with patch(
        "app.graph.clinical_graph.extract_assessment_with_confidence",
        return_value=(make_assessment(), low_confidence),
    ):
        from app.graph.clinical_graph import clinical_graph

        result = clinical_graph.invoke({
            "transcript": "Test clinical transcript"
        })

    assert "error" in result
    assert result["error"]["message"] == (
        "Extraction confidence is below the required threshold."
    )

    low_fields = result["error"]["fields"]

    assert any(
        field["field"] == "patientAdvice"
        and field["confidence"] == 0.50
        for field in low_fields
    )
