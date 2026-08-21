from app.schemas.assessment import FirstAssessment
from pydantic import ValidationError
import pytest

def test_defaults():
    fa = FirstAssessment()
    assert fa.clinicalDetails.clinicalHistory == ""
    assert fa.subjectiveAssessments == []
    assert fa.objectiveAssessment.tests == []
    assert fa.patientAdvice.adviceDetails == ""

def test_extra_fields_forbidden():
    with pytest.raises(ValidationError):
        FirstAssessment(extra_field="not allowed")

def test_serialization_keys():
    fa = FirstAssessment()
    keys = set(fa.model_dump().keys())
    expected = {"clinicalDetails", "subjectiveAssessments", "objectiveAssessment",
                "subjectiveGoals", "objectiveGoals", "recommendation", "patientAdvice"}
    assert keys == expected