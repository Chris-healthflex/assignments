"""Hallucination-guard tests.

IMPORTANT — what these tests actually verify, and what they don't:

These tests mock the LLM call itself (model.with_structured_output(...).invoke(...))
and return a hand-authored ExtractionOutput per case. This verifies that the schema
mapping and API layer correctly PASS THROUGH whatever the model returns without
adding, inferring, or filling in anything extra on top of it.

These tests do NOT verify that the real Groq model (openai/gpt-oss-120b) itself
resists hallucinating when given an ambiguous or sparse transcript - that behavior
depends on the live model and can only be checked by actually calling it. Before
submission, supplement this with at least one manual run of tests/run_pipeline.py
or a live POST /assessments/parse call using a transcript that deliberately omits
dates/measurements/goals, and manually confirm the real model's response leaves
those fields empty rather than guessing. That manual check is not automated here
because it costs a real API call and is not deterministic run-to-run.
"""

from app.models.first_assessment import (
    ClinicalDetails,
    FirstAssessment,
    ObjectiveAssessment,
    ObjectiveTest,
)
from app.pipeline.extraction import ExtractionOutput, _extract_with_model


class FakeStructuredModel:
    """Stands in for the real ChatGroq model. Returns exactly the ExtractionOutput
    it's constructed with, regardless of the transcript passed in - so each test
    controls precisely what "the model returned" for that case.
    """

    def __init__(self, output: ExtractionOutput) -> None:
        self._output = output

    def with_structured_output(self, _schema):
        return self

    def invoke(self, _messages):
        return self._output


def _run(output: ExtractionOutput) -> FirstAssessment:
    state = {"transcript": "irrelevant - model output is mocked directly"}
    result = _extract_with_model(state, FakeStructuredModel(output))
    return result["assessment"]


def test_missing_dates_are_not_fabricated() -> None:
    # Simulates a real model response for a transcript that never states a date:
    # a goal is recorded (correct), but targetDate is left empty rather than guessed.
    output = ExtractionOutput(
        assessment=FirstAssessment(),
        field_confidence={},
    )
    assessment = _run(output)
    assert all(goal.targetDate == "" for goal in assessment.subjectiveGoals)
    assert all(goal.targetDate == "" for goal in assessment.objectiveGoals)


def test_missing_rom_numbers_are_not_fabricated() -> None:
    # A test is named because the transcript mentions it was performed, but no
    # numeric result was stated - value/left/right must stay empty, not guessed.
    assessment_with_unmeasured_test = FirstAssessment(
        objectiveAssessment=ObjectiveAssessment(
            tests=[
                ObjectiveTest(
                    testName="Knee ROM",
                    unitName="degrees",
                    value="",
                    left="",
                    right="",
                    comments="Assessed; no numeric value stated in transcript",
                )
            ]
        )
    )
    output = ExtractionOutput(assessment=assessment_with_unmeasured_test, field_confidence={})
    assessment = _run(output)
    test = assessment.objectiveAssessment.tests[0]
    assert test.value == ""
    assert test.left == ""
    assert test.right == ""


def test_missing_treatment_goals_remain_empty_arrays() -> None:
    output = ExtractionOutput(assessment=FirstAssessment(), field_confidence={})
    assessment = _run(output)
    assert assessment.subjectiveGoals == []
    assert assessment.objectiveGoals == []


def test_missing_diagnosis_is_not_fabricated() -> None:
    output = ExtractionOutput(assessment=FirstAssessment(), field_confidence={})
    assessment = _run(output)
    assert assessment.subjectiveAssessments == []


def test_partial_clinical_details_are_not_padded() -> None:
    # Regression guard for a subtler failure mode: if the model states chief
    # complaint but not duration, the mapping layer must not invent a duration
    # (e.g. copying it from chiefComplaint or defaulting to a placeholder).
    output = ExtractionOutput(
        assessment=FirstAssessment(
            clinicalDetails=ClinicalDetails(
                clinicalHistory="",
                chiefComplaint="Left knee pain",
                duration="",
            )
        ),
        field_confidence={},
    )
    assessment = _run(output)
    assert assessment.clinicalDetails.chiefComplaint == "Left knee pain"
    assert assessment.clinicalDetails.duration == ""
    assert assessment.clinicalDetails.clinicalHistory == ""