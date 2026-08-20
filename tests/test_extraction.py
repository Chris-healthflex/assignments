import pytest
from app.agents.clinical_extraction_graph import (
    create_clinical_extraction_graph,
    validate_transcript_node,
    validate_extraction_node,
    build_first_assessment_node,
    ExtractionState,
)
from app.services.extraction import ClinicalExtractionService
from app.core.errors import ExtractionError


def test_validate_transcript_node_rejects_empty():
    """Verify validate_transcript node flags empty transcripts."""
    state: ExtractionState = {
        "transcript": "",
        "extracted_data": {},
        "validation_errors": [],
        "confidence_score": 1.0,
        "first_assessment": None,
        "is_valid": True,
    }
    result = validate_transcript_node(state)
    assert result["is_valid"] is False
    assert any("empty" in e.lower() for e in result["validation_errors"])


def test_validate_transcript_node_rejects_short_noise():
    """Verify validate_transcript node flags uninformative short transcripts."""
    state: ExtractionState = {
        "transcript": "Hello doctor",
        "extracted_data": {},
        "validation_errors": [],
        "confidence_score": 1.0,
        "first_assessment": None,
        "is_valid": True,
    }
    result = validate_transcript_node(state)
    assert result["is_valid"] is False
    assert any("too brief" in e.lower() for e in result["validation_errors"])


def test_anti_hallucination_validation_flags_unsupported_facts():
    """Verify validate_extraction node flags hallucinated facts not found in transcript."""
    transcript = "The patient reports knee pain for approximately three weeks."
    extracted_data = {
        "clinicalDetails": {
            "clinicalHistory": "chronic osteoarthritis of spine",
            "chiefComplaint": "knee pain",
            "duration": "3 years",  # Hallucinated duration (spoken was three weeks)
        },
        "subjectiveAssessments": [],
        "objectiveAssessment": {
            "tests": [
                {
                    "testName": "Blood Pressure",
                    "unitName": "mmHg",
                    "value": "120/80",  # Hallucinated measurement
                    "left": "",
                    "right": "",
                    "comments": "",
                }
            ]
        },
        "subjectiveGoals": [],
        "objectiveGoals": [],
        "recommendation": [],
        "patientAdvice": {"adviceDetails": ""},
    }

    state: ExtractionState = {
        "transcript": transcript,
        "extracted_data": extracted_data,
        "validation_errors": [],
        "confidence_score": 1.0,
        "first_assessment": None,
        "is_valid": True,
    }

    result = validate_extraction_node(state)
    assert result["confidence_score"] < 0.70
    assert result["is_valid"] is False
    assert any("below threshold" in e.lower() for e in result["validation_errors"])


def test_anti_hallucination_passes_grounded_data():
    """Verify validate_extraction node accepts grounded data."""
    transcript = (
        "The patient reports knee pain for approximately three weeks. "
        "Left knee flexion was 120 degrees."
    )
    extracted_data = {
        "clinicalDetails": {
            "clinicalHistory": "",
            "chiefComplaint": "knee pain",
            "duration": "three weeks",
        },
        "subjectiveAssessments": [],
        "objectiveAssessment": {
            "tests": [
                {
                    "testName": "flexion",
                    "unitName": "degrees",
                    "value": "120",
                    "left": "120",
                    "right": "",
                    "comments": "",
                }
            ]
        },
        "subjectiveGoals": [],
        "objectiveGoals": [],
        "recommendation": [],
        "patientAdvice": {"adviceDetails": ""},
    }

    state: ExtractionState = {
        "transcript": transcript,
        "extracted_data": extracted_data,
        "validation_errors": [],
        "confidence_score": 1.0,
        "first_assessment": None,
        "is_valid": True,
    }

    result = validate_extraction_node(state)
    assert result["confidence_score"] >= 0.70
    assert result["is_valid"] is True


@pytest.mark.asyncio
async def test_extraction_service_end_to_end():
    """Verify ClinicalExtractionService executes extraction on realistic clinical text."""
    transcript = (
        "Patient comes in complaining of right shoulder pain for 2 weeks after tennis. "
        "Flexion measures 110 degrees on the right side. "
        "Goal is to play tennis again by December 1. "
        "Recommended weekly physiotherapy and advised daily shoulder stretching."
    )

    service = ClinicalExtractionService()
    assessment = await service.extract_assessment(transcript)

    assert assessment is not None
    assert "shoulder pain" in assessment.clinicalDetails.chiefComplaint.lower()
    assert "2 weeks" in assessment.clinicalDetails.duration.lower()
    assert isinstance(assessment.objectiveAssessment.tests, list)
    assert isinstance(assessment.recommendation, list)


@pytest.mark.asyncio
async def test_extraction_service_raises_on_empty():
    """Verify ClinicalExtractionService raises ExtractionError on empty transcript."""
    service = ClinicalExtractionService()
    with pytest.raises(ExtractionError):
        await service.extract_assessment("")
