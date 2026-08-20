import pytest
from app.agents.clinical_extraction_graph import (
    validate_extraction_node,
    _check_field_grounding,
    ExtractionState,
)
from app.schemas.first_assessment import FirstAssessment


def test_field_grounding_matches_exact_content():
    """Verify that field words found in the transcript pass grounding check."""
    transcript = "Patient has severe knee pain after running 5 kilometers."
    is_ok, warning = _check_field_grounding("clinicalDetails.chiefComplaint", "severe knee pain", transcript)
    assert is_ok is True
    assert warning is None


def test_field_grounding_rejects_hallucinated_content():
    """Verify that words not mentioned in the transcript fail grounding check."""
    transcript = "Patient has knee pain after running."
    is_ok, warning = _check_field_grounding("objectiveAssessment.tests[0].value", "Blood Pressure 180/120 mmHg", transcript)
    assert is_ok is False
    assert "lacks verbatim evidence" in warning


def test_validate_extraction_node_prunes_synthetic_test_names():
    """Verify that subjectiveAssessments with ungrounded synthetic test names are pruned."""
    transcript = "Doctor: Take some rest. Patient: I have loose stools and cramps."
    state: ExtractionState = {
        "transcript": transcript,
        "extracted_data": {
            "clinicalDetails": {
                "clinicalHistory": "",
                "chiefComplaint": "loose stools and cramps",
                "duration": ""
            },
            "subjectiveAssessments": [
                {
                    "testName": "Synthetic Fabricated Stool Test",
                    "conclusion": "Positive"
                }
            ],
            "objectiveAssessment": {"tests": []},
            "subjectiveGoals": [],
            "objectiveGoals": [],
            "recommendation": [
                {"sessionType": "Take some rest", "sessionFrequency": ""}
            ],
            "patientAdvice": {"adviceDetails": "Rest"}
        },
        "validation_errors": [],
        "confidence_score": 1.0,
        "first_assessment": None,
        "is_valid": True,
    }

    result = validate_extraction_node(state)
    cleaned = result["extracted_data"]
    # The synthetic test name should be pruned
    assert len(cleaned["subjectiveAssessments"]) == 0
    # The valid recommendation and complaint should remain
    assert len(cleaned["recommendation"]) == 1
    assert cleaned["clinicalDetails"]["chiefComplaint"] == "loose stools and cramps"


def test_speaker_attribution_requires_doctor_for_recommendations():
    """Verify that recommendations with evidence only in patient text (not doctor) are pruned."""
    transcript = (
        "Doctor: Hello, how are you? "
        "Patient: I am drinking lots of fluids and took ibuprofen yesterday."
    )
    state: ExtractionState = {
        "transcript": transcript,
        "extracted_data": {
            "clinicalDetails": {
                "clinicalHistory": "",
                "chiefComplaint": "drinking lots of fluids",
                "duration": ""
            },
            "subjectiveAssessments": [],
            "objectiveAssessment": {"tests": []},
            "subjectiveGoals": [],
            "objectiveGoals": [],
            "recommendation": [
                {"sessionType": "Continue drinking fluids", "sessionFrequency": ""}
            ],
            "patientAdvice": {"adviceDetails": ""}
        },
        "validation_errors": [],
        "confidence_score": 1.0,
        "first_assessment": None,
        "is_valid": True,
    }

    result = validate_extraction_node(state)
    cleaned = result["extracted_data"]
    # Recommendation should be pruned because doctor did not say it
    assert len(cleaned["recommendation"]) == 0
