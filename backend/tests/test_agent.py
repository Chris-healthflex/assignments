"""Tests for the extraction graph, driven by a stubbed model.

Every test is offline and deterministic: the "model" is a list of canned
extractions handed out one call at a time, which also lets the tests assert how
many calls the graph made.
"""

import pytest

from clinical_assessment.agent import (
    MAX_REPAIR_ATTEMPTS,
    AssessmentOutcome,
    run_agent,
)
from clinical_assessment.extraction_models import (
    ExtractedValue,
    RawClinicalDetails,
    RawExtraction,
    RawObjectiveAssessment,
    RawObjectiveTest,
    RawPatientAdvice,
)

TRANSCRIPT = (
    "Clinician: What brings you in? Patient: My right shoulder has been painful "
    "for about three weeks. Clinician: Abduction is 120 degrees on the right."
)


def value(text: str = "", evidence: str = "") -> ExtractedValue:
    """Build an ExtractedValue, defaulting to the "not stated" pair."""
    return ExtractedValue(value=text, evidence=evidence)


def extraction(test: RawObjectiveTest | None = None) -> RawExtraction:
    """Build an extraction that is blank apart from an optional objective test."""
    return RawExtraction(
        clinicalDetails=RawClinicalDetails(
            clinicalHistory=value(), chiefComplaint=value(), duration=value()
        ),
        subjectiveAssessments=[],
        objectiveAssessment=RawObjectiveAssessment(
            tests=[] if test is None else [test]
        ),
        subjectiveGoals=[],
        objectiveGoals=[],
        recommendation=[],
        patientAdvice=RawPatientAdvice(adviceDetails=value()),
    )


def objective_test(measurement: ExtractedValue) -> RawObjectiveTest:
    """Build an objective test carrying one measurement."""
    return RawObjectiveTest(
        testName=value("Abduction", "Abduction is 120 degrees"),
        unitName=value(),
        value=measurement,
        left=value(),
        right=value(),
        comments=value(),
    )


GROUNDED = objective_test(value("120", "Abduction is 120 degrees"))
FABRICATED = objective_test(value("45", "Abduction is 120 degrees"))


class StubModel:
    """Hands out canned extractions, recording how many calls were made."""

    def __init__(self, *responses: RawExtraction) -> None:
        self._responses = list(responses)
        self.prompts: list[str] = []

    def __call__(self, prompt: str) -> RawExtraction:
        self.prompts.append(prompt)
        # The last response repeats, so an "always fails" stub never runs dry.
        index = min(len(self.prompts) - 1, len(self._responses) - 1)
        return self._responses[index]


def test_fully_grounded_extraction_finalises_without_repair() -> None:
    model = StubModel(extraction(GROUNDED))

    outcome = run_agent(TRANSCRIPT, model)

    assert (outcome.attempts, len(model.prompts)) == (0, 1)


def test_fully_grounded_extraction_is_reported_at_full_confidence() -> None:
    model = StubModel(extraction(GROUNDED))

    outcome = run_agent(TRANSCRIPT, model)

    assert outcome.confidence == 1.0


def test_one_repair_round_recovers_an_ungrounded_field() -> None:
    model = StubModel(extraction(FABRICATED), extraction(GROUNDED))

    outcome = run_agent(TRANSCRIPT, model)

    assert (outcome.attempts, outcome.confidence) == (1, 1.0)


def test_repair_recovers_the_value_into_the_assessment() -> None:
    model = StubModel(extraction(FABRICATED), extraction(GROUNDED))

    outcome = run_agent(TRANSCRIPT, model)

    assert outcome.assessment.objectiveAssessment.tests[0].value == "120"


def test_repair_narrows_the_prompt_to_the_failed_fields_only() -> None:
    model = StubModel(extraction(FABRICATED), extraction(GROUNDED))

    run_agent(TRANSCRIPT, model)

    repair_prompt = model.prompts[1]
    assert [line for line in repair_prompt.splitlines() if line.startswith("- ")] == [
        "- objectiveAssessment.tests[0].value: quoted 'Abduction is 120 degrees'"
    ]


def test_repair_budget_exhausted_returns_low_confidence_not_an_exception() -> None:
    model = StubModel(extraction(FABRICATED))

    outcome = run_agent(TRANSCRIPT, model)

    assert isinstance(outcome, AssessmentOutcome) and outcome.confidence < 1.0


def test_graph_terminates_when_repair_never_succeeds() -> None:
    model = StubModel(extraction(FABRICATED))

    outcome = run_agent(TRANSCRIPT, model)

    # One first pass plus MAX_REPAIR_ATTEMPTS repairs, and no more.
    assert (len(model.prompts), outcome.attempts) == (
        MAX_REPAIR_ATTEMPTS + 1,
        MAX_REPAIR_ATTEMPTS,
    )


def test_unrepairable_field_is_blanked_in_the_final_assessment() -> None:
    model = StubModel(extraction(FABRICATED))

    outcome = run_agent(TRANSCRIPT, model)

    assert outcome.assessment.objectiveAssessment.tests[0].value == ""


def test_unrepairable_field_is_reported_to_the_caller() -> None:
    model = StubModel(extraction(FABRICATED))

    outcome = run_agent(TRANSCRIPT, model)

    assert [verdict.field_path for verdict in outcome.ungrounded_fields] == [
        "objectiveAssessment.tests[0].value"
    ]


def test_empty_transcript_produces_all_blank_fields() -> None:
    model = StubModel(extraction())

    outcome = run_agent("", model)

    dumped = outcome.assessment.model_dump()
    assert dumped["clinicalDetails"] == {
        "clinicalHistory": "",
        "chiefComplaint": "",
        "duration": "",
    }


def test_model_failure_propagates_to_the_caller() -> None:
    def failing_model(prompt: str) -> RawExtraction:
        raise RuntimeError("model exploded")

    with pytest.raises(RuntimeError, match="model exploded"):
        run_agent(TRANSCRIPT, failing_model)
