import pytest
from pydantic import ValidationError

from app.schemas.first_assessment import FirstAssessment, normalize_assessment


def test_valid_first_assessment():
    assessment = FirstAssessment(
        clinicalDetails={
            "clinicalHistory": "Prior knee pain.",
            "chiefComplaint": "Right knee pain",
            "duration": "2 weeks",
        },
        subjectiveAssessments=[{"testName": "Pain report", "conclusion": "Pain with stairs"}],
        objectiveAssessment={
            "tests": [
                {
                    "testName": "Knee flexion",
                    "unitName": "degrees",
                    "value": "120",
                    "left": "",
                    "right": "120",
                    "comments": "",
                }
            ]
        },
        subjectiveGoals=[{"goalDetails": "Walk without pain", "targetDate": ""}],
        objectiveGoals=[
            {
                "goalName": "Knee flexion",
                "goalCategory": "ROM",
                "unitName": "degrees",
                "value": "130",
                "targetDate": "",
            }
        ],
        recommendation=[{"sessionType": "Physiotherapy", "sessionFrequency": "twice weekly"}],
        patientAdvice={"adviceDetails": "Continue home exercises."},
    )

    assert assessment.clinicalDetails.chiefComplaint == "Right knee pain"
    assert assessment.objectiveAssessment.tests[0].right == "120"


def test_missing_string_values_become_empty_strings():
    assessment = normalize_assessment({"clinicalDetails": {"chiefComplaint": None}})

    assert assessment.clinicalDetails.chiefComplaint == ""
    assert assessment.clinicalDetails.clinicalHistory == ""
    assert assessment.patientAdvice.adviceDetails == ""


def test_missing_list_values_become_empty_lists():
    assessment = normalize_assessment(
        {
            "subjectiveAssessments": None,
            "subjectiveGoals": None,
            "objectiveGoals": None,
            "recommendation": None,
        }
    )

    assert assessment.subjectiveAssessments == []
    assert assessment.subjectiveGoals == []
    assert assessment.objectiveGoals == []
    assert assessment.recommendation == []


def test_extra_fields_are_rejected():
    with pytest.raises(ValidationError):
        FirstAssessment.model_validate({"unexpected": "field"})


def test_invalid_nested_data_is_rejected():
    with pytest.raises(ValidationError):
        FirstAssessment.model_validate({"clinicalDetails": "not an object"})
