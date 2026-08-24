"""The FirstAssessment schema is the contract — lock its shape."""
import json
from app.schemas.assessment import (
    FirstAssessment, ObjectiveTest, SCALAR_FIELD_PATHS, LIST_FIELD_PATHS,
)
import pytest
from pydantic import ValidationError


def test_empty_assessment_has_all_sections():
    a = FirstAssessment()
    d = a.model_dump()
    for key in ("clinicalDetails", "subjectiveAssessments", "objectiveAssessment",
                "subjectiveGoals", "objectiveGoals", "recommendation", "patientAdvice"):
        assert key in d


def test_objective_test_has_left_right_columns():
    fields = ObjectiveTest.model_fields.keys()
    assert {"testName", "unitName", "value", "left", "right", "comments"} <= set(fields)


def test_extra_keys_rejected():
    with pytest.raises(ValidationError):
        FirstAssessment(unexpected="x")


def test_none_coerced_to_empty_string():
    t = ObjectiveTest(testName=None, left=124, right=None)
    assert t.testName == "" and t.left == "124" and t.right == ""


def test_sample_output_validates_against_schema():
    with open("data/sample_output.json") as f:
        data = json.load(f)
    FirstAssessment.model_validate(data["assessment"])  # must not raise


def test_field_path_constants_are_consistent():
    assert "clinicalDetails.duration" in SCALAR_FIELD_PATHS
    assert "objectiveAssessment.tests" in LIST_FIELD_PATHS
