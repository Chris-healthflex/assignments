"""Contract tests for the FirstAssessment schema.

These encode the brief's exact wording independently of the implementation.
The expected shape below is transcribed from the assignment rather than
generated from the model, so a rename or an accidental extra field fails here
instead of reaching the frontend.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.schemas.first_assessment import (
    FirstAssessment,
    SECTION_KEYS,
    blank_field_paths,
    empty_assessment,
    iter_field_paths,
)

# Transcribed directly from the brief: 7 sections, exact key names.
EXPECTED_SHAPE: dict[str, object] = {
    "clinicalDetails": {"clinicalHistory", "chiefComplaint", "duration"},
    "subjectiveAssessments": [{"testName", "conclusion"}],
    "objectiveAssessment": {
        "tests": [{"testName", "unitName", "value", "left", "right", "comments"}]
    },
    "subjectiveGoals": [{"goalDetails", "targetDate"}],
    "objectiveGoals": [{"goalName", "goalCategory", "unitName", "value", "targetDate"}],
    "recommendation": [{"sessionType", "sessionFrequency"}],
    "patientAdvice": {"adviceDetails"},
}


# --------------------------------------------------------------------------
# Structure
# --------------------------------------------------------------------------
def test_has_exactly_seven_sections():
    assert len(SECTION_KEYS) == 7
    assert set(SECTION_KEYS) == set(EXPECTED_SHAPE)


def test_top_level_keys_match_brief_exactly():
    """No extra fields, no renamed keys, no missing keys."""
    dumped = empty_assessment().model_dump()
    assert list(dumped) == list(EXPECTED_SHAPE), "Top-level keys drifted from the brief"


@pytest.mark.parametrize("section", ["clinicalDetails", "patientAdvice"])
def test_object_section_keys(section):
    dumped = empty_assessment().model_dump()
    assert set(dumped[section]) == EXPECTED_SHAPE[section]


def test_objective_assessment_wraps_a_tests_array():
    """objectiveAssessment is an object whose only key is the tests array."""
    dumped = empty_assessment().model_dump()
    assert set(dumped["objectiveAssessment"]) == {"tests"}
    assert dumped["objectiveAssessment"]["tests"] == []


@pytest.mark.parametrize(
    "section,expected_keys",
    [
        ("subjectiveAssessments", EXPECTED_SHAPE["subjectiveAssessments"][0]),
        ("subjectiveGoals", EXPECTED_SHAPE["subjectiveGoals"][0]),
        ("objectiveGoals", EXPECTED_SHAPE["objectiveGoals"][0]),
        ("recommendation", EXPECTED_SHAPE["recommendation"][0]),
    ],
)
def test_array_item_keys(section, expected_keys):
    """Populate one item and confirm its keys match the brief."""
    item = {key: "x" for key in expected_keys}
    dumped = FirstAssessment.model_validate({section: [item]}).model_dump()
    assert set(dumped[section][0]) == expected_keys


def test_objective_test_item_keys():
    keys = EXPECTED_SHAPE["objectiveAssessment"]["tests"][0]
    payload = {"objectiveAssessment": {"tests": [{k: "x" for k in keys}]}}
    dumped = FirstAssessment.model_validate(payload).model_dump()
    assert set(dumped["objectiveAssessment"]["tests"][0]) == keys


# --------------------------------------------------------------------------
# Rule 1 - no extra fields, no renamed keys
# --------------------------------------------------------------------------
def test_unknown_top_level_key_is_rejected():
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        FirstAssessment.model_validate({"confidence": 0.9})


def test_renamed_key_is_rejected():
    """A plausible typo must fail loudly, not pass through silently."""
    with pytest.raises(ValidationError):
        # Note the trailing 's' - a rename the frontend would not understand.
        FirstAssessment.model_validate({"clinicalDetails": {"chiefComplaints": "knee pain"}})


def test_confidence_metadata_cannot_be_smuggled_into_a_section():
    """S5 flags must live outside FirstAssessment, not inside a section."""
    with pytest.raises(ValidationError):
        FirstAssessment.model_validate(
            {"patientAdvice": {"adviceDetails": "rest", "confidence": 0.4}}
        )


# --------------------------------------------------------------------------
# Rule 2 - array fields are always arrays
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "section",
    ["subjectiveAssessments", "subjectiveGoals", "objectiveGoals", "recommendation"],
)
def test_null_array_becomes_empty_array(section):
    dumped = FirstAssessment.model_validate({section: None}).model_dump()
    assert dumped[section] == []


def test_single_object_is_wrapped_into_an_array():
    """The brief requires arrays even when only one item is present."""
    dumped = FirstAssessment.model_validate(
        {"recommendation": {"sessionType": "Physiotherapy", "sessionFrequency": "twice weekly"}}
    ).model_dump()
    assert isinstance(dumped["recommendation"], list)
    assert len(dumped["recommendation"]) == 1
    assert dumped["recommendation"][0]["sessionType"] == "Physiotherapy"


def test_nested_tests_array_survives_null():
    dumped = FirstAssessment.model_validate({"objectiveAssessment": {"tests": None}}).model_dump()
    assert dumped["objectiveAssessment"]["tests"] == []


# --------------------------------------------------------------------------
# Rule 3 - string fields are strings, never null
# --------------------------------------------------------------------------
def test_null_string_becomes_empty_string():
    dumped = FirstAssessment.model_validate(
        {"clinicalDetails": {"chiefComplaint": None, "duration": None}}
    ).model_dump()
    assert dumped["clinicalDetails"]["chiefComplaint"] == ""
    assert dumped["clinicalDetails"]["duration"] == ""


def test_null_object_section_becomes_blank_object():
    dumped = FirstAssessment.model_validate({"patientAdvice": None}).model_dump()
    assert dumped["patientAdvice"] == {"adviceDetails": ""}


def test_no_nulls_anywhere_in_a_fully_null_payload():
    """Worst case: every section null. Output must still be frontend-safe."""
    payload = {key: None for key in SECTION_KEYS}
    dumped = FirstAssessment.model_validate(payload).model_dump()

    def assert_no_nulls(node, path="root"):
        if node is None:
            pytest.fail(f"null found at {path}")
        if isinstance(node, dict):
            for key, val in node.items():
                assert_no_nulls(val, f"{path}.{key}")
        elif isinstance(node, list):
            for index, val in enumerate(node):
                assert_no_nulls(val, f"{path}[{index}]")

    assert_no_nulls(dumped)


def test_numeric_value_is_coerced_to_string():
    """An LLM reading '120 degrees' often emits the bare number 120."""
    dumped = FirstAssessment.model_validate(
        {"objectiveAssessment": {"tests": [{"testName": "Knee flexion", "value": 120}]}}
    ).model_dump()
    assert dumped["objectiveAssessment"]["tests"][0]["value"] == "120"


def test_whitespace_is_trimmed():
    dumped = FirstAssessment.model_validate(
        {"clinicalDetails": {"chiefComplaint": "  knee pain \n"}}
    ).model_dump()
    assert dumped["clinicalDetails"]["chiefComplaint"] == "knee pain"


# --------------------------------------------------------------------------
# Round-trip and field-path helpers
# --------------------------------------------------------------------------
def test_round_trip_is_stable():
    """Serialise then revalidate must be a no-op - this is what MongoDB does."""
    original = FirstAssessment.model_validate(
        {
            "clinicalDetails": {"chiefComplaint": "right knee pain", "duration": "3 weeks"},
            "recommendation": [{"sessionType": "Physiotherapy", "sessionFrequency": "2x/week"}],
            "objectiveAssessment": {
                "tests": [{"testName": "Flexion", "value": "120", "unitName": "deg"}]
            },
        }
    )
    dumped = original.model_dump()
    assert FirstAssessment.model_validate(dumped).model_dump() == dumped


def test_empty_assessment_has_no_populated_leaves():
    assessment = empty_assessment()
    paths = list(iter_field_paths(assessment))
    # 3 from clinicalDetails + 1 from patientAdvice; empty arrays contribute none.
    assert len(paths) == 4
    assert len(blank_field_paths(assessment)) == 4


def test_defaults_are_not_shared_between_instances():
    """Class-level mutable defaults must be deep-copied, not aliased.

    If they were shared, one parsed assessment would leak goals and complaints
    into the next request served by the same process.
    """
    first, second = empty_assessment(), empty_assessment()
    first.subjectiveGoals.append({"goalDetails": "walk unaided", "targetDate": ""})
    first.clinicalDetails.chiefComplaint = "knee pain"

    assert second.subjectiveGoals == []
    assert second.clinicalDetails.chiefComplaint == ""


def test_contract_survives_runtime_mutation():
    """validate_assignment keeps the no-null rule true after construction."""
    assessment = empty_assessment()
    assessment.clinicalDetails.chiefComplaint = None
    assert assessment.clinicalDetails.chiefComplaint == ""

    assessment.subjectiveGoals = None
    assert assessment.subjectiveGoals == []


def test_field_paths_address_array_items_precisely():
    assessment = FirstAssessment.model_validate(
        {
            "clinicalDetails": {"chiefComplaint": "knee pain"},
            "objectiveGoals": [
                {"goalName": "Restore flexion", "targetDate": ""},
                {"goalName": "", "targetDate": "2026-09-01"},
            ],
        }
    )
    paths = dict(iter_field_paths(assessment))

    assert paths["clinicalDetails.chiefComplaint"] == "knee pain"
    assert paths["objectiveGoals[0].goalName"] == "Restore flexion"
    assert paths["objectiveGoals[1].targetDate"] == "2026-09-01"

    blanks = blank_field_paths(assessment)
    assert "objectiveGoals[0].targetDate" in blanks
    assert "objectiveGoals[1].goalName" in blanks
    assert "clinicalDetails.chiefComplaint" not in blanks
