import pytest
from pydantic import ValidationError
from app.models.assessment import (
    FirstAssessment,
    ClinicalDetails,
    SubjectiveAssessment,
    ObjectiveAssessment,
    ObjectiveTest,
    SubjectiveGoal,
    ObjectiveGoal,
    Recommendation,
    PatientAdvice,
)


def test_default_instantiation_has_no_nulls():
    assessment = FirstAssessment()
    data = assessment.model_dump()

    # Check top-level keys
    assert isinstance(data["clinicalDetails"], dict)
    assert data["clinicalDetails"]["clinicalHistory"] == ""
    assert data["clinicalDetails"]["chiefComplaint"] == ""
    assert data["clinicalDetails"]["duration"] == ""

    assert data["subjectiveAssessments"] == []
    assert data["objectiveAssessment"]["tests"] == []
    assert data["subjectiveGoals"] == []
    assert data["objectiveGoals"] == []
    assert data["recommendation"] == []
    assert data["patientAdvice"]["adviceDetails"] == ""

    # Verify recursively that no value is None
    def assert_no_null(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                assert v is not None, f"Key {k} is None"
                assert_no_null(v)
        elif isinstance(obj, list):
            for item in obj:
                assert item is not None
                assert_no_null(item)

    assert_no_null(data)


def test_extra_field_forbidden():
    # Extra field at root
    with pytest.raises(ValidationError):
        FirstAssessment(unknown_field="invalid")

    # Extra field in nested model
    with pytest.raises(ValidationError):
        FirstAssessment(clinicalDetails={"chiefComplaint": "Knee pain", "extraKey": 123})

    # Extra field in array item
    with pytest.raises(ValidationError):
        FirstAssessment(
            objectiveAssessment={
                "tests": [{"testName": "ROM", "bogus": "value"}]
            }
        )


def test_null_coercion():
    # Null string coerced to ""
    cd = ClinicalDetails(clinicalHistory=None, chiefComplaint=None, duration=None)
    assert cd.clinicalHistory == ""
    assert cd.chiefComplaint == ""
    assert cd.duration == ""

    # Null list coerced to []
    oa = ObjectiveAssessment(tests=None)
    assert oa.tests == []

    # Null nested models coerced to default empty models
    fa = FirstAssessment(
        clinicalDetails=None,
        subjectiveAssessments=None,
        objectiveAssessment=None,
        subjectiveGoals=None,
        objectiveGoals=None,
        recommendation=None,
        patientAdvice=None,
    )
    assert fa.clinicalDetails.chiefComplaint == ""
    assert fa.subjectiveAssessments == []
    assert fa.objectiveAssessment.tests == []
    assert fa.subjectiveGoals == []
    assert fa.objectiveGoals == []
    assert fa.recommendation == []
    assert fa.patientAdvice.adviceDetails == ""


def test_whitespace_stripping():
    cd = ClinicalDetails(chiefComplaint="  Left knee ACL tear  ")
    assert cd.chiefComplaint == "Left knee ACL tear"
