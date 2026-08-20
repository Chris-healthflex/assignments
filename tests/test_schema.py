import pytest
from pydantic import ValidationError
from app.schemas.first_assessment import (
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


def test_first_assessment_default_instantiation():
    """Verify that empty instantiation produces valid non-null structures and arrays."""
    assessment = FirstAssessment()
    data = assessment.model_dump()

    # Verify top-level structure
    assert "clinicalDetails" in data
    assert "subjectiveAssessments" in data
    assert "objectiveAssessment" in data
    assert "subjectiveGoals" in data
    assert "objectiveGoals" in data
    assert "recommendation" in data
    assert "patientAdvice" in data

    # Verify clinicalDetails
    assert data["clinicalDetails"]["clinicalHistory"] == ""
    assert data["clinicalDetails"]["chiefComplaint"] == ""
    assert data["clinicalDetails"]["duration"] == ""

    # Verify array fields are lists
    assert isinstance(data["subjectiveAssessments"], list)
    assert isinstance(data["objectiveAssessment"]["tests"], list)
    assert isinstance(data["subjectiveGoals"], list)
    assert isinstance(data["objectiveGoals"], list)
    assert isinstance(data["recommendation"], list)

    # Verify patientAdvice
    assert data["patientAdvice"]["adviceDetails"] == ""


def test_first_assessment_converts_none_to_empty_string():
    """Verify that None values for string fields are automatically converted to empty strings."""
    raw_payload = {
        "clinicalDetails": {
            "clinicalHistory": None,
            "chiefComplaint": None,
            "duration": None,
        },
        "subjectiveAssessments": [
            {"testName": None, "conclusion": None}
        ],
        "objectiveAssessment": {
            "tests": [
                {
                    "testName": None,
                    "unitName": None,
                    "value": None,
                    "left": None,
                    "right": None,
                    "comments": None,
                }
            ]
        },
        "subjectiveGoals": [
            {"goalDetails": None, "targetDate": None}
        ],
        "objectiveGoals": [
            {
                "goalName": None,
                "goalCategory": None,
                "unitName": None,
                "value": None,
                "targetDate": None,
            }
        ],
        "recommendation": [
            {"sessionType": None, "sessionFrequency": None}
        ],
        "patientAdvice": {
            "adviceDetails": None
        }
    }

    assessment = FirstAssessment.model_validate(raw_payload)
    dumped = assessment.model_dump()

    # None fields must be empty strings
    assert dumped["clinicalDetails"]["clinicalHistory"] == ""
    assert dumped["clinicalDetails"]["chiefComplaint"] == ""
    assert dumped["clinicalDetails"]["duration"] == ""

    assert dumped["subjectiveAssessments"][0]["testName"] == ""
    assert dumped["subjectiveAssessments"][0]["conclusion"] == ""

    assert dumped["objectiveAssessment"]["tests"][0]["testName"] == ""
    assert dumped["objectiveAssessment"]["tests"][0]["unitName"] == ""
    assert dumped["objectiveAssessment"]["tests"][0]["value"] == ""

    assert dumped["patientAdvice"]["adviceDetails"] == ""


def test_first_assessment_coerces_single_object_to_array():
    """Verify that if an object is mistakenly provided instead of an array, it is coerced to a list."""
    raw_payload = {
        "subjectiveAssessments": {"testName": "VAS", "conclusion": "6/10"},
        "recommendation": {"sessionType": "Physiotherapy", "sessionFrequency": "2x/week"},
    }

    assessment = FirstAssessment.model_validate(raw_payload)
    assert isinstance(assessment.subjectiveAssessments, list)
    assert len(assessment.subjectiveAssessments) == 1
    assert assessment.subjectiveAssessments[0].testName == "VAS"

    assert isinstance(assessment.recommendation, list)
    assert len(assessment.recommendation) == 1
    assert assessment.recommendation[0].sessionType == "Physiotherapy"


def test_first_assessment_forbids_extra_fields():
    """Verify that undeclared extra fields raise a ValidationError."""
    with pytest.raises(ValidationError):
        FirstAssessment.model_validate({
            "extra_field": "not_allowed"
        })

    with pytest.raises(ValidationError):
        ClinicalDetails.model_validate({
            "chiefComplaint": "Knee pain",
            "unsupportedField": "extra"
        })


def test_first_assessment_exact_casing(sample_first_assessment: FirstAssessment):
    """Verify exact casing matches frontend expectations."""
    dumped = sample_first_assessment.model_dump()
    expected_keys = {
        "clinicalDetails",
        "subjectiveAssessments",
        "objectiveAssessment",
        "subjectiveGoals",
        "objectiveGoals",
        "recommendation",
        "patientAdvice",
    }
    assert set(dumped.keys()) == expected_keys
    assert set(dumped["clinicalDetails"].keys()) == {"clinicalHistory", "chiefComplaint", "duration"}
    assert set(dumped["patientAdvice"].keys()) == {"adviceDetails"}


def test_first_assessment_rejects_snake_case_renamed_fields():
    """Verify that snake_case renamed fields (e.g. chief_complaint) are rejected."""
    with pytest.raises(ValidationError):
        FirstAssessment.model_validate({
            "clinicalDetails": {
                "chief_complaint": "Knee pain"
            }
        })


def test_first_assessment_array_variations_0_1_2_items():
    """Verify 0, 1, and 2 item arrays serialize correctly."""
    # 0 items
    a0 = FirstAssessment(recommendation=[])
    assert a0.recommendation == []

    # 1 item
    a1 = FirstAssessment(recommendation=[Recommendation(sessionType="Fluids")])
    assert len(a1.recommendation) == 1
    assert a1.recommendation[0].sessionType == "Fluids"

    # 2 items
    a2 = FirstAssessment(recommendation=[
        Recommendation(sessionType="Fluids"),
        Recommendation(sessionType="Rest", sessionFrequency="2 days")
    ])
    assert len(a2.recommendation) == 2
    assert a2.recommendation[1].sessionFrequency == "2 days"

