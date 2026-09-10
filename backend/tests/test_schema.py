"""Conformance tests for the FirstAssessment production contract."""

from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

from clinical_assessment.schema import (
    ClinicalDetails,
    FirstAssessment,
    ObjectiveAssessment,
    ObjectiveGoal,
    ObjectiveTest,
    PatientAdvice,
    Recommendation,
    SubjectiveAssessment,
    SubjectiveGoal,
)

# Transcribed from the brief and owned by the test, not by the code under test:
# production may not quietly drift the contract into agreement with itself.
# Verified against the brief's Step 3 "FirstAssessment Schema" panel
# (screenshot, 2026-09-10): all 21 paths match, including the singular-named
# `recommendation` array and the `tests` wrapper under `objectiveAssessment`.
PRODUCTION_KEY_PATHS: frozenset[str] = frozenset(
    {
        "clinicalDetails.clinicalHistory",
        "clinicalDetails.chiefComplaint",
        "clinicalDetails.duration",
        "subjectiveAssessments[].testName",
        "subjectiveAssessments[].conclusion",
        "objectiveAssessment.tests[].testName",
        "objectiveAssessment.tests[].unitName",
        "objectiveAssessment.tests[].value",
        "objectiveAssessment.tests[].left",
        "objectiveAssessment.tests[].right",
        "objectiveAssessment.tests[].comments",
        "subjectiveGoals[].goalDetails",
        "subjectiveGoals[].targetDate",
        "objectiveGoals[].goalName",
        "objectiveGoals[].goalCategory",
        "objectiveGoals[].unitName",
        "objectiveGoals[].value",
        "objectiveGoals[].targetDate",
        "recommendation[].sessionType",
        "recommendation[].sessionFrequency",
        "patientAdvice.adviceDetails",
    }
)

CLINICAL_DETAILS_FIELDS: dict[str, Any] = {
    "clinicalHistory": "No prior surgery",
    "chiefComplaint": "Right shoulder pain",
    "duration": "3 weeks",
}
SUBJECTIVE_ASSESSMENT_FIELDS: dict[str, Any] = {
    "testName": "Pain scale",
    "conclusion": "Pain reported on overhead reach",
}
OBJECTIVE_TEST_FIELDS: dict[str, Any] = {
    "testName": "Shoulder abduction ROM",
    "unitName": "degrees",
    "value": "120",
    "left": "150",
    "right": "120",
    "comments": "Measured with a goniometer",
}
SUBJECTIVE_GOAL_FIELDS: dict[str, Any] = {
    "goalDetails": "Sleep through the night without pain",
    "targetDate": "2026-11-01",
}
OBJECTIVE_GOAL_FIELDS: dict[str, Any] = {
    "goalName": "Restore abduction",
    "goalCategory": "Range of motion",
    "unitName": "degrees",
    "value": "160",
    "targetDate": "2026-12-01",
}
RECOMMENDATION_FIELDS: dict[str, Any] = {
    "sessionType": "Physiotherapy",
    "sessionFrequency": "Twice weekly",
}
PATIENT_ADVICE_FIELDS: dict[str, Any] = {"adviceDetails": "Apply ice after exercises"}
FIRST_ASSESSMENT_FIELDS: dict[str, Any] = {
    "clinicalDetails": CLINICAL_DETAILS_FIELDS,
    "subjectiveAssessments": [SUBJECTIVE_ASSESSMENT_FIELDS],
    "objectiveAssessment": {"tests": [OBJECTIVE_TEST_FIELDS]},
    "subjectiveGoals": [SUBJECTIVE_GOAL_FIELDS],
    "objectiveGoals": [OBJECTIVE_GOAL_FIELDS],
    "recommendation": [RECOMMENDATION_FIELDS],
    "patientAdvice": PATIENT_ADVICE_FIELDS,
}

VALID_FIELDS_BY_MODEL: dict[type[BaseModel], dict[str, Any]] = {
    ClinicalDetails: CLINICAL_DETAILS_FIELDS,
    SubjectiveAssessment: SUBJECTIVE_ASSESSMENT_FIELDS,
    ObjectiveTest: OBJECTIVE_TEST_FIELDS,
    ObjectiveAssessment: {"tests": [OBJECTIVE_TEST_FIELDS]},
    SubjectiveGoal: SUBJECTIVE_GOAL_FIELDS,
    ObjectiveGoal: OBJECTIVE_GOAL_FIELDS,
    Recommendation: RECOMMENDATION_FIELDS,
    PatientAdvice: PATIENT_ADVICE_FIELDS,
    FirstAssessment: FIRST_ASSESSMENT_FIELDS,
}

MODEL_CASES = [
    pytest.param(model, id=model.__name__) for model in VALID_FIELDS_BY_MODEL
]
FIELD_CASES = [
    pytest.param(model, field_name, id=f"{model.__name__}.{field_name}")
    for model, fields in VALID_FIELDS_BY_MODEL.items()
    for field_name in fields
]
# Only str-typed leaves can carry the "not null" guarantee; the nested-model and
# list fields of FirstAssessment are covered through their own models above.
STRING_FIELD_CASES = [
    pytest.param(model, field_name, id=f"{model.__name__}.{field_name}")
    for model, fields in VALID_FIELDS_BY_MODEL.items()
    for field_name, value in fields.items()
    if isinstance(value, str)
]


def flatten_key_paths(node: Any, prefix: str = "") -> set[str]:
    """Flatten a dumped model into dotted leaf paths, marking lists with ``[]``.

    Args:
        node: A value from ``model_dump()`` — mapping, list, or leaf.
        prefix: The path accumulated so far; empty at the root.

    Returns:
        The set of leaf paths reachable in ``node``. An empty list contributes
        no paths, which is why the conformance fixture keeps one item in every
        array.
    """
    if isinstance(node, dict):
        return {
            path
            for key, child in node.items()
            for path in flatten_key_paths(child, f"{prefix}.{key}" if prefix else key)
        }
    if isinstance(node, list):
        return {
            path for item in node for path in flatten_key_paths(item, f"{prefix}[]")
        }
    return {prefix}


def test_dump_produces_the_exact_production_key_set() -> None:
    assessment = FirstAssessment(**FIRST_ASSESSMENT_FIELDS)

    dumped = assessment.model_dump()

    # Set equality catches a missing key and an extra key in one assertion.
    assert flatten_key_paths(dumped) == set(PRODUCTION_KEY_PATHS)


def test_section_with_one_item_still_serialises_as_array() -> None:
    assessment = FirstAssessment(**FIRST_ASSESSMENT_FIELDS)

    dumped = assessment.model_dump()

    assert dumped["recommendation"] == [RECOMMENDATION_FIELDS]


def test_empty_section_serialises_as_empty_array() -> None:
    fields = FIRST_ASSESSMENT_FIELDS | {"subjectiveAssessments": []}

    dumped = FirstAssessment(**fields).model_dump()

    assert dumped["subjectiveAssessments"] == []


@pytest.mark.parametrize("model", MODEL_CASES)
def test_rejects_unknown_field(model: type[BaseModel]) -> None:
    fields = VALID_FIELDS_BY_MODEL[model] | {"notAContractField": "x"}

    with pytest.raises(ValidationError):
        model(**fields)


@pytest.mark.parametrize(("model", "field_name"), STRING_FIELD_CASES)
def test_rejects_null_for_string_field(model: type[BaseModel], field_name: str) -> None:
    fields = VALID_FIELDS_BY_MODEL[model] | {field_name: None}

    with pytest.raises(ValidationError):
        model(**fields)


@pytest.mark.parametrize(("model", "field_name"), FIELD_CASES)
def test_rejects_missing_required_field(
    model: type[BaseModel], field_name: str
) -> None:
    fields = {
        key: value
        for key, value in VALID_FIELDS_BY_MODEL[model].items()
        if key != field_name
    }

    with pytest.raises(ValidationError):
        model(**fields)
