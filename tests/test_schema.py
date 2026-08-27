import pytest
from pydantic import ValidationError
from app.models.schema import (
    FirstAssessment,
    ClinicalDetails,
    SubjectiveAssessment,
    ObjectiveTest,
    ObjectiveAssessment,
    SubjectiveGoal,
    ObjectiveGoal,
    Recommendation,
    PatientAdvice,
)


def test_first_assessment_exact_structure():
    """Verify standard creation with non-null string fields and arrays"""
    assessment = FirstAssessment(
        clinicalDetails=ClinicalDetails(
            clinicalHistory="Patient injured back lifting heavy machinery.",
            chiefComplaint="Lumbar back pain",
            duration="3 weeks",
        ),
        subjectiveAssessments=[
            SubjectiveAssessment(
                testName="Lumbar Assessment",
                conclusion="Acute lumbosacral radiculopathy",
            )
        ],
        objectiveAssessment=ObjectiveAssessment(
            tests=[
                ObjectiveTest(
                    testName="Lumbar Flexion",
                    unitName="degrees",
                    value="45 degrees",
                    left="",
                    right="",
                    comments="Restricted with pain",
                )
            ]
        ),
        subjectiveGoals=[
            SubjectiveGoal(
                goalDetails="Return to work without pain",
                targetDate="4 weeks",
            )
        ],
        objectiveGoals=[
            ObjectiveGoal(
                goalName="Increase flexion",
                goalCategory="ROM",
                unitName="degrees",
                value="80 degrees",
                targetDate="6 weeks",
            )
        ],
        recommendation=[
            Recommendation(
                sessionType="Physical Therapy",
                sessionFrequency="2x per week",
            )
        ],
        patientAdvice=PatientAdvice(
            adviceDetails="Apply ice and avoid heavy loads."
        ),
    )

    dump = assessment.model_dump()
    # Ensure all 7 sections are present
    assert "clinicalDetails" in dump
    assert "subjectiveAssessments" in dump
    assert "objectiveAssessment" in dump
    assert "subjectiveGoals" in dump
    assert "objectiveGoals" in dump
    assert "recommendation" in dump
    assert "patientAdvice" in dump

    # Check non-null string enforcement
    for k, v in dump["clinicalDetails"].items():
        assert isinstance(v, str)
        assert v is not None

    # Check array enforcement
    assert isinstance(dump["subjectiveAssessments"], list)
    assert isinstance(dump["objectiveAssessment"]["tests"], list)
    assert isinstance(dump["subjectiveGoals"], list)
    assert isinstance(dump["objectiveGoals"], list)
    assert isinstance(dump["recommendation"], list)


def test_schema_forbids_extra_fields():
    """Ensure extra keys are strictly rejected (extra='forbid')"""
    with pytest.raises(ValidationError):
        FirstAssessment.model_validate({
            "clinicalDetails": {
                "clinicalHistory": "test",
                "chiefComplaint": "test",
                "duration": "1 week",
                "extraField": "forbidden",
            },
            "subjectiveAssessments": [],
            "objectiveAssessment": {"tests": []},
            "subjectiveGoals": [],
            "objectiveGoals": [],
            "recommendation": [],
            "patientAdvice": {"adviceDetails": ""},
        })


def test_null_coercion_to_empty_string():
    """Ensure null inputs in string fields are coerced to empty strings without crashing"""
    details = ClinicalDetails.model_validate({
        "clinicalHistory": None,
        "chiefComplaint": "Back pain",
        "duration": None,
    })
    assert details.clinicalHistory == ""
    assert details.chiefComplaint == "Back pain"
    assert details.duration == ""
