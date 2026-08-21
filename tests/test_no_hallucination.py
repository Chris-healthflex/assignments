from app.models.first_assessment import FirstAssessment
from app.pipeline.extraction import ExtractionOutput, _extract_with_model
from tests.fixtures.assessment_cases import assessment_from_transcript


class FakeStructuredModel:
    def __init__(self, transcript_to_assessment):
        self.transcript_to_assessment = transcript_to_assessment
        self.transcript = ""

    def with_structured_output(self, _schema):
        return self

    def invoke(self, messages):
        self.transcript = messages[-1][1].removeprefix("Transcript:\n")
        return ExtractionOutput(
            assessment=self.transcript_to_assessment(self.transcript),
            field_confidence={},
        )


def extract(transcript: str) -> FirstAssessment:
    result = _extract_with_model(
        {"transcript": transcript},
        FakeStructuredModel(assessment_from_transcript),
    )
    return result["assessment"]


def test_missing_dates_are_not_fabricated() -> None:
    assessment = extract("The patient has knee pain and should improve walking. No dates are discussed.")
    assert all(goal.targetDate == "" for goal in assessment.subjectiveGoals)
    assert all(goal.targetDate == "" for goal in assessment.objectiveGoals)


def test_missing_rom_numbers_are_not_fabricated() -> None:
    assessment = extract("Knee ROM was assessed, but no numeric ROM values were stated.")
    test = assessment.objectiveAssessment.tests[0]
    assert test.value == ""
    assert test.left == ""
    assert test.right == ""


def test_missing_treatment_goals_remain_empty_arrays() -> None:
    assessment = extract("The patient reports knee pain. No treatment goals were mentioned.")
    assert assessment.subjectiveGoals == []
    assert assessment.objectiveGoals == []


def test_missing_diagnosis_is_not_fabricated() -> None:
    assessment = extract("The patient reports knee pain. No explicit diagnosis is stated.")
    assert assessment.subjectiveAssessments == []
