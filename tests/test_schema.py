import pytest
from pydantic import ValidationError

from app.models import FirstAssessment

SECTIONS = {
    "clinicalDetails",
    "subjectiveAssessments",
    "objectiveAssessment",
    "subjectiveGoals",
    "objectiveGoals",
    "recommendation",
    "patientAdvice",
}


def test_schema_has_exactly_the_required_shape():
    data = FirstAssessment().model_dump()

    assert set(data) == SECTIONS
    assert set(data["clinicalDetails"]) == {
        "clinicalHistory",
        "chiefComplaint",
        "duration",
    }
    assert set(data["objectiveAssessment"]) == {"tests"}
    assert set(data["patientAdvice"]) == {"adviceDetails"}
    assert data["subjectiveAssessments"] == []


def test_nulls_become_empty_strings():
    data = FirstAssessment.model_validate(
        {
            "clinicalDetails": {"clinicalHistory": None},
            "patientAdvice": {"adviceDetails": None},
        }
    ).model_dump()

    assert data["clinicalDetails"]["clinicalHistory"] == ""
    assert data["patientAdvice"]["adviceDetails"] == ""


def test_a_single_object_is_kept_as_an_array():
    data = FirstAssessment.model_validate(
        {"recommendation": {"sessionType": "Physiotherapy", "sessionFrequency": "Weekly"}}
    ).model_dump()

    assert data["recommendation"] == [
        {"sessionType": "Physiotherapy", "sessionFrequency": "Weekly"}
    ]


def test_extra_and_renamed_fields_are_rejected():
    with pytest.raises(ValidationError):
        FirstAssessment.model_validate({"patientName": "Jane Doe"})
    with pytest.raises(ValidationError):
        FirstAssessment.model_validate({"clinicalDetails": {"onset": "8 months"}})


def test_objective_test_values_are_strings():
    test = (
        FirstAssessment.model_validate(
            {
                "objectiveAssessment": {
                    "tests": [{"testName": "Knee flexion", "left": "124"}]
                }
            }
        )
        .model_dump()["objectiveAssessment"]["tests"][0]
    )

    assert set(test) == {"testName", "unitName", "value", "left", "right", "comments"}
    assert all(isinstance(value, str) for value in test.values())
