"""Unit tests for strict FirstAssessment Pydantic v2 schemas."""

import json
import pytest
from pydantic import ValidationError

from app.schemas.assessment import (
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


def test_first_assessment_exact_top_level_keys():
    """Test 1: FirstAssessment has exact expected top-level keys."""
    expected_keys = {
        "clinicalDetails",
        "subjectiveAssessments",
        "objectiveAssessment",
        "subjectiveGoals",
        "objectiveGoals",
        "recommendation",
        "patientAdvice",
    }
    assessment = FirstAssessment()
    dumped = assessment.model_dump()
    assert set(dumped.keys()) == expected_keys


def test_first_assessment_rejects_extra_top_level_fields():
    """Test 2: Extra top-level fields are rejected by extra='forbid'."""
    with pytest.raises(ValidationError) as exc_info:
        FirstAssessment.model_validate({"extraField": "unexpected", "confidence": 0.95})
    errors = exc_info.value.errors()
    assert any(err["loc"] == ("extraField",) and err["type"] == "extra_forbidden" for err in errors)


def test_nested_models_reject_extra_fields():
    """Test 3: Nested models reject unexpected fields."""
    with pytest.raises(ValidationError):
        ClinicalDetails.model_validate({"clinicalHistory": "Healed scar", "unapprovedKey": "bad"})

    with pytest.raises(ValidationError):
        ObjectiveGoal.model_validate({
            "goalName": "ROM",
            "goalCategory": "Mobility",
            "unitName": "deg",
            "value": "110",
            "targetDate": "2026-10-01",
            "score": 100,  # extra field
        })


def test_objective_assessment_rejects_top_level_comments():
    """Test 4: objectiveAssessment rejects an unapproved top-level comments field."""
    with pytest.raises(ValidationError) as exc_info:
        ObjectiveAssessment.model_validate({
            "tests": [],
            "comments": ["This should not be at the objectiveAssessment level"],
        })
    errors = exc_info.value.errors()
    assert any(err["loc"] == ("comments",) and err["type"] == "extra_forbidden" for err in errors)


def test_objective_test_comments_is_an_array():
    """Test 5: ObjectiveTest.comments is an array and behaves properly."""
    # Default is empty list
    test_item = ObjectiveTest(testName="Knee Flexion", unitName="degrees", value="110")
    assert isinstance(test_item.comments, list)
    assert test_item.comments == []

    # Single comment
    test_single = ObjectiveTest(
        testName="Knee Flexion",
        unitName="degrees",
        value="110",
        comments=["Pain at end range"],
    )
    assert test_single.comments == ["Pain at end range"]
    assert isinstance(test_single.model_dump()["comments"], list)


def test_subjective_assessment_conclusion_is_an_array():
    """Test 6: subjectiveAssessments[].conclusion is an array."""
    sub = SubjectiveAssessment(testName="Pain Scale", conclusion=["Moderate pain with walking"])
    dumped = sub.model_dump()
    assert isinstance(dumped["conclusion"], list)
    assert len(dumped["conclusion"]) == 1
    assert dumped["conclusion"][0] == "Moderate pain with walking"


def test_array_fields_remain_arrays_when_empty():
    """Test 7: All required array fields default to empty lists and serialize as arrays."""
    assessment = FirstAssessment()
    dumped = assessment.model_dump()

    assert isinstance(dumped["subjectiveAssessments"], list)
    assert dumped["subjectiveAssessments"] == []

    assert isinstance(dumped["objectiveAssessment"]["tests"], list)
    assert dumped["objectiveAssessment"]["tests"] == []

    assert isinstance(dumped["subjectiveGoals"], list)
    assert dumped["subjectiveGoals"] == []

    assert isinstance(dumped["objectiveGoals"], list)
    assert dumped["objectiveGoals"] == []

    assert isinstance(dumped["recommendation"], list)
    assert dumped["recommendation"] == []


def test_array_fields_remain_arrays_with_one_element():
    """Test 8: Array fields remain lists when containing exactly one element."""
    assessment = FirstAssessment(
        subjectiveAssessments=[SubjectiveAssessment(testName="Mobility", conclusion=["Restricted"])],
        objectiveAssessment=ObjectiveAssessment(tests=[ObjectiveTest(testName="Extension", value="0")]),
        subjectiveGoals=[SubjectiveGoal(goalDetails="Walk 1km", targetDate="2026-10-01")],
        objectiveGoals=[ObjectiveGoal(goalName="Flexion", goalCategory="ROM", unitName="deg", value="120", targetDate="2026-10-01")],
        recommendation=[Recommendation(sessionType="Physiotherapy", sessionFrequency="1x/week for 4 sessions")],
    )
    dumped = assessment.model_dump()

    assert isinstance(dumped["subjectiveAssessments"], list)
    assert len(dumped["subjectiveAssessments"]) == 1

    assert isinstance(dumped["objectiveAssessment"]["tests"], list)
    assert len(dumped["objectiveAssessment"]["tests"]) == 1

    assert isinstance(dumped["subjectiveGoals"], list)
    assert len(dumped["subjectiveGoals"]) == 1

    assert isinstance(dumped["objectiveGoals"], list)
    assert len(dumped["objectiveGoals"]) == 1

    assert isinstance(dumped["recommendation"], list)
    assert len(dumped["recommendation"]) == 1


def test_string_fields_do_not_serialize_as_null():
    """Test 9: String fields default to empty strings and never serialize as None/null."""
    assessment = FirstAssessment()
    dumped = assessment.model_dump()

    # Verify root nested objects string fields
    assert dumped["clinicalDetails"]["clinicalHistory"] == ""
    assert dumped["clinicalDetails"]["chiefComplaint"] == ""
    assert dumped["patientAdvice"]["adviceDetails"] == ""

    # Dump to JSON string and verify no "null" values exist in JSON
    json_str = assessment.model_dump_json()
    parsed_json = json.loads(json_str)
    assert parsed_json["clinicalDetails"]["clinicalHistory"] is not None
    assert parsed_json["clinicalDetails"]["chiefComplaint"] is not None
    assert parsed_json["patientAdvice"]["adviceDetails"] is not None


def test_model_dump_produces_json_compatible_data():
    """Test 10: model_dump() produces JSON-compatible structure with no serialization errors."""
    assessment = FirstAssessment(
        clinicalDetails=ClinicalDetails(
            clinicalHistory="Road traffic accident 8 months ago, left tibial condyle fracture, ORIF.",
            chiefComplaint="Left knee pain and walking difficulty.",
            duration={"text": "8 months"},
        ),
        subjectiveAssessments=[
            SubjectiveAssessment(testName="Pain Assessment", conclusion=["Moderate pain", "Mild irritability"])
        ],
        objectiveAssessment=ObjectiveAssessment(
            tests=[
                ObjectiveTest(
                    testName="Knee Flexion ROM",
                    unitName="degrees",
                    value="110",
                    left="110",
                    right="135",
                    comments=["Painful at end range"],
                )
            ]
        ),
        subjectiveGoals=[
            SubjectiveGoal(goalDetails="Prolonged walking without back pain", targetDate="2026-10-15")
        ],
        objectiveGoals=[
            ObjectiveGoal(
                goalName="Knee Flexion",
                goalCategory="Range of Motion",
                unitName="degrees",
                value="130",
                targetDate="2026-10-15",
            )
        ],
        recommendation=[
            Recommendation(sessionType="Physiotherapy", sessionFrequency="Once weekly for 4 sessions")
        ],
        patientAdvice=PatientAdvice(
            adviceDetails="Avoid prolonged standing; continue progressive quadriceps loading exercises."
        ),
    )

    dumped = assessment.model_dump()
    json_str = assessment.model_dump_json()
    round_trip = json.loads(json_str)

    assert round_trip == dumped
    assert round_trip["clinicalDetails"]["duration"] == {"text": "8 months"}
    assert round_trip["objectiveAssessment"]["tests"][0]["comments"] == ["Painful at end range"]
