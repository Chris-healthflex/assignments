import pytest
from app.services.extraction_agent import ClinicalExtractionAgent
from app.models.schema import FirstAssessment


def test_extraction_from_clinical_transcript():
    transcript = (
        "Clinician: Good morning. What brings you in today?\n"
        "Patient: I have had lower back pain for 3 weeks after lifting heavy boxes.\n"
        "Clinician: Physical examination reveals lumbar flexion is 45 degrees. Straight leg raise on left is 40 degrees and right is 75 degrees. Tenderness over L4-L5.\n"
        "Clinician: Assessment shows acute lumbar strain. Objective goal is to achieve 80 degrees lumbar flexion in 6 weeks.\n"
        "Clinician: Subjective goal is walking without pain within 4 weeks. I recommend physical therapy twice weekly.\n"
        "Clinician: Patient advice is to apply ice and avoid heavy lifting."
    )

    result = ClinicalExtractionAgent.run(transcript)
    assert result["success"] is True
    assert result["assessment"] is not None
    assert isinstance(result["assessment"], FirstAssessment)

    assessment: FirstAssessment = result["assessment"]
    assert "pain" in assessment.clinicalDetails.chiefComplaint.lower() or "back" in assessment.clinicalDetails.chiefComplaint.lower()
    assert "3 weeks" in assessment.clinicalDetails.duration.lower()
    assert len(assessment.objectiveAssessment.tests) > 0
    assert len(assessment.subjectiveGoals) > 0
    assert len(assessment.objectiveGoals) > 0
    assert len(assessment.recommendation) > 0
    assert len(assessment.patientAdvice.adviceDetails) > 0
    assert result["confidence"].overall_score >= 0.5


def test_extraction_flags_empty_or_non_clinical_text():
    gibberish = "123 456 789"
    result = ClinicalExtractionAgent.run(gibberish)
    assert result["success"] is False
    assert result["assessment"] is None
    assert len(result["validation_errors"]) > 0
