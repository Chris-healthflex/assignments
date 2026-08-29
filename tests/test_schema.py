import pytest
from pydantic import ValidationError

from app.schemas.assessment import (
    ClinicalDetails,
    FirstAssessment,
    ObjectiveAssessment,
    PatientAdvice,
)


def _minimal_assessment_kwargs() -> dict:
    return dict(
        clinicalDetails=ClinicalDetails(
            clinicalHistory="", chiefComplaint="Knee pain", duration="3 weeks"
        ),
        subjectiveAssessments=[],
        objectiveAssessment=ObjectiveAssessment(tests=[]),
        subjectiveGoals=[],
        objectiveGoals=[],
        recommendation=[],
        patientAdvice=PatientAdvice(adviceDetails=""),
    )


def test_valid_assessment_constructs():
    assessment = FirstAssessment(**_minimal_assessment_kwargs())
    assert assessment.clinicalDetails.chiefComplaint == "Knee pain"


def test_array_fields_are_always_lists():
    assessment = FirstAssessment(**_minimal_assessment_kwargs())
    dumped = assessment.model_dump()
    assert isinstance(dumped["subjectiveAssessments"], list)
    assert isinstance(dumped["objectiveGoals"], list)
    assert isinstance(dumped["recommendation"], list)


def test_extra_fields_are_rejected():
    kwargs = _minimal_assessment_kwargs()
    kwargs["confidence"] = 0.95  # not part of the schema
    with pytest.raises(ValidationError):
        FirstAssessment(**kwargs)


def test_string_fields_cannot_be_null():
    kwargs = _minimal_assessment_kwargs()
    kwargs["clinicalDetails"] = ClinicalDetails(
        clinicalHistory="", chiefComplaint="Knee pain", duration="3 weeks"
    )
    # Directly attempting None should fail validation
    with pytest.raises(ValidationError):
        ClinicalDetails(clinicalHistory=None, chiefComplaint="x", duration="y")
