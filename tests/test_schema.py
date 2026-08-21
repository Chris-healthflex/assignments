from typing import get_origin

from app.models.first_assessment import (
    FirstAssessment,
    ObjectiveGoal,
    ObjectiveTest,
    Recommendation,
    SubjectiveAssessment,
    SubjectiveGoal,
)


def test_first_assessment_has_exact_top_level_sections() -> None:
    assessment = FirstAssessment()
    assert set(assessment.model_dump()) == {
        "clinicalDetails",
        "subjectiveAssessments",
        "objectiveAssessment",
        "subjectiveGoals",
        "objectiveGoals",
        "recommendation",
        "patientAdvice",
    }
    assert assessment.subjectiveAssessments == []
    assert assessment.objectiveAssessment.tests == []


def test_first_assessment_matches_production_nested_keys_and_types() -> None:
    assessment = FirstAssessment()
    assert set(assessment.clinicalDetails.model_dump()) == {
        "clinicalHistory",
        "chiefComplaint",
        "duration",
    }
    assert set(assessment.patientAdvice.model_dump()) == {"adviceDetails"}
    assert set(ObjectiveTest().model_dump()) == {
        "testName",
        "unitName",
        "value",
        "left",
        "right",
        "comments",
    }
    assert set(SubjectiveAssessment().model_dump()) == {"testName", "conclusion"}
    assert set(SubjectiveGoal().model_dump()) == {"goalDetails", "targetDate"}
    assert set(ObjectiveGoal().model_dump()) == {
        "goalName",
        "goalCategory",
        "unitName",
        "value",
        "targetDate",
    }
    assert set(Recommendation().model_dump()) == {"sessionType", "sessionFrequency"}

    assert get_origin(FirstAssessment.model_fields["subjectiveAssessments"].annotation) is list
    assert get_origin(FirstAssessment.model_fields["subjectiveGoals"].annotation) is list
    assert get_origin(FirstAssessment.model_fields["objectiveGoals"].annotation) is list
    assert get_origin(FirstAssessment.model_fields["recommendation"].annotation) is list


def test_empty_schema_serializes_strings_instead_of_nulls() -> None:
    payload = FirstAssessment().model_dump()
    assert payload["clinicalDetails"] == {
        "clinicalHistory": "",
        "chiefComplaint": "",
        "duration": "",
    }
    assert payload["patientAdvice"] == {"adviceDetails": ""}


def test_unknown_schema_fields_are_rejected() -> None:
    try:
        FirstAssessment(unexpected="value")
    except Exception as exc:
        assert "unexpected" in str(exc)
    else:
        raise AssertionError("Unknown fields must be rejected")
