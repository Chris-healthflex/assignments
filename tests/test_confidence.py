"""Unit tests for grounding validation and anti-hallucination checks."""

import pytest
from app.schemas.assessment import (
    ClinicalDetails,
    FirstAssessment,
    ObjectiveAssessment,
    ObjectiveGoal,
    ObjectiveTest,
    SubjectiveGoal,
)
from app.services.confidence import validate_grounding


@pytest.fixture
def sample_clinical_transcript() -> str:
    """Representative clinical transcript excerpt."""
    return """
    Patient is an active individual who had a road traffic accident eight months ago.
    Sustained left tibial condyle fracture and avulsion ACL tear, underwent ORIF.
    Current chief complaint is left knee pain during prolonged walking and standing.
    Objective examination:
    Knee flexion measured left 124 degrees and right 130 degrees.
    Knee extension left 20 degrees, right -5 degrees.
    Ankle dorsiflexion left 4.5 degrees, right 12 degrees.
    Plan: Physiotherapy once weekly for four sessions.
    """


def test_grounding_passes_for_supported_measurements(sample_clinical_transcript: str):
    """Test that measurements explicitly present in transcript are verified as grounded."""
    assessment = FirstAssessment(
        clinicalDetails=ClinicalDetails(
            chiefComplaint="Left knee pain",
            clinicalHistory="Road traffic accident eight months ago",
        ),
        objectiveAssessment=ObjectiveAssessment(
            tests=[
                ObjectiveTest(testName="Knee Flexion", unitName="degrees", left="124", right="130"),
                ObjectiveTest(testName="Ankle Dorsiflexion", unitName="degrees", left="4.5", right="12"),
            ]
        ),
    )

    result = validate_grounding(sample_clinical_transcript, assessment)
    assert result.is_grounded is True
    assert len(result.uncertain_fields) == 0
    assert len(result.evidence) >= 4  # matched values recorded in evidence


def test_grounding_flags_hallucinated_measurement(sample_clinical_transcript: str):
    """Test that an ungrounded/hallucinated numeric value is flagged in uncertain_fields."""
    assessment = FirstAssessment(
        objectiveAssessment=ObjectiveAssessment(
            tests=[
                # 199 is NOT in the transcript
                ObjectiveTest(testName="Knee Flexion", unitName="degrees", left="199", right="130")
            ]
        )
    )

    result = validate_grounding(sample_clinical_transcript, assessment)
    assert result.is_grounded is False
    assert len(result.uncertain_fields) == 1
    assert "199" in result.uncertain_fields[0]["value"]
    assert "not found or supported" in result.uncertain_fields[0]["reason"]


def test_grounding_flags_hallucinated_target_date(sample_clinical_transcript: str):
    """Test that a calendar date not mentioned in transcript is flagged as uncertain."""
    assessment = FirstAssessment(
        subjectiveGoals=[
            SubjectiveGoal(goalDetails="Return to running", targetDate="2026-12-31")
        ]
    )

    result = validate_grounding(sample_clinical_transcript, assessment)
    assert len(result.uncertain_fields) >= 1
    assert any("2026-12-31" in item["value"] for item in result.uncertain_fields)


def test_empty_target_dates_and_values_are_valid(sample_clinical_transcript: str):
    """Test that leaving unmentioned target dates and goal values empty does not trigger uncertainty."""
    assessment = FirstAssessment(
        subjectiveGoals=[],
        objectiveGoals=[
            ObjectiveGoal(goalName="Knee Extension and stability", value="", unitName="", targetDate="")
        ],
    )

    result = validate_grounding(sample_clinical_transcript, assessment)
    assert not any("targetDate" in item.get("field", "") for item in result.uncertain_fields)
    assert not any("value" in item.get("field", "") for item in result.uncertain_fields)


def test_grounding_flags_hallucinated_goal_value(sample_clinical_transcript: str):
    """Test that an invented objective goal target number not in transcript is flagged."""
    assessment = FirstAssessment(
        objectiveGoals=[
            # '0' is not spoken in the transcript as a goal target
            ObjectiveGoal(goalName="Knee Extension", value="0", unitName="degrees", targetDate="")
        ]
    )

    result = validate_grounding(sample_clinical_transcript, assessment)
    assert result.is_grounded is False
    assert len(result.uncertain_fields) >= 1
    assert any("objectiveGoals[0].value" == item.get("field") for item in result.uncertain_fields)
