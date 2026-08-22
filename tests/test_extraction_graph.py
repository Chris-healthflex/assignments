"""End-to-end tests for the LangGraph extraction agent.

The LLM is stubbed, so these run in milliseconds with no Ollama daemon, no GPU
and no model download - which is what lets them run in CI.
"""

from __future__ import annotations

import json

from app.extraction.graph import _normalise_tests, extract_assessment
from app.schemas.first_assessment import SECTION_KEYS


TRANSCRIPT = (
    "The patient presented with left knee pain following surgery eight months "
    "ago after a road traffic accident. Left knee flexion was 124° compared "
    "with 130° on the right. Physiotherapy was recommended once weekly for "
    "four sessions."
)


class StubLLM:
    """Returns a canned payload per section, keyed off the prompt text."""

    def __init__(
        self,
        overrides: dict[str, dict] | None = None,
        fail: set[str] | None = None,
    ):
        self.overrides = overrides or {}
        self.fail = fail or set()
        self.calls: list[str] = []

    def _section_for(self, prompt: str) -> str:
        """Identify the section from the complete user prompt.

        Match the distinctive phrases used by the real extraction prompts.
        Order matters: objective must be checked before the generic fallback
        to plan.
        """
        prompt_lower = prompt.lower()

        if "clinical background" in prompt_lower:
            return "clinicalDetails"

        if "what the patient reported" in prompt_lower:
            return "subjective"

        if "objective measurements" in prompt_lower:
            return "objective"

        if "treatment goals" in prompt_lower:
            return "goals"

        if "treatment plan" in prompt_lower:
            return "plan"

        raise AssertionError(
            f"StubLLM could not identify section from prompt:\n{prompt}"
        )

    def invoke(self, messages):
        # structured_call sends the user prompt as the second message.
        prompt = messages[1][1]
        section = self._section_for(prompt)
        self.calls.append(section)

        if section in self.fail:

            class Broken:
                content = "I'm sorry, I cannot help with that."

            return Broken()

        defaults = {
            "clinicalDetails": {
                "clinicalHistory": "road traffic accident",
                "chiefComplaint": "left knee pain",
                "duration": "eight months",
            },
            "subjective": {
                "subjectiveAssessments": [
                    {
                        "testName": "Pain",
                        "conclusion": "left knee pain",
                    }
                ]
            },
            "objective": {
                "tests": [
                    {
                        "testName": "Knee flexion",
                        "unitName": "degrees",
                        "left": "124",
                        "right": "130",
                        "value": "",
                        "comments": "",
                    }
                ]
            },
            "goals": {
                "subjectiveGoals": [],
                "objectiveGoals": [],
            },
            "plan": {
                "recommendation": [
                    {
                        "sessionType": "Physiotherapy",
                        "sessionFrequency": "once weekly for four sessions",
                    }
                ],
                "patientAdvice": {
                    "adviceDetails": "",
                },
            },
        }

        body = self.overrides.get(section, defaults[section])

        class Response:
            content = json.dumps(body)

        return Response()


# --------------------------------------------------------------------------
# Normalisation
# --------------------------------------------------------------------------


def test_normalise_sided_measurement_clears_value():
    tests = [
        {
            "testName": "Knee flexion",
            "unitName": "degrees",
            "value": "124",
            "left": "124",
            "right": "130",
            "comments": "",
        }
    ]

    result = _normalise_tests(tests)

    assert result[0]["left"] == "124"
    assert result[0]["right"] == "130"
    assert result[0]["value"] == ""


def test_normalise_single_value_measurement_keeps_value():
    tests = [
        {
            "testName": "Pain score",
            "unitName": "",
            "value": "5",
            "left": "",
            "right": "",
            "comments": "",
        }
    ]

    result = _normalise_tests(tests)

    assert result[0]["value"] == "5"
    assert result[0]["left"] == ""
    assert result[0]["right"] == ""


# --------------------------------------------------------------------------
# Happy path
# --------------------------------------------------------------------------


def test_graph_produces_a_schema_valid_assessment():
    result = extract_assessment(TRANSCRIPT, llm=StubLLM())
    dumped = result["assessment"].model_dump()

    assert list(dumped) == list(SECTION_KEYS)

    assert dumped["clinicalDetails"]["chiefComplaint"] == "left knee pain"

    assert dumped["objectiveAssessment"]["tests"][0]["left"] == "124"

    assert dumped["objectiveAssessment"]["tests"][0]["right"] == "130"

    assert dumped["recommendation"][0]["sessionType"] == "Physiotherapy"


def test_every_section_node_runs():
    stub = StubLLM()

    extract_assessment(TRANSCRIPT, llm=stub)

    assert stub.calls == [
        "clinicalDetails",
        "subjective",
        "objective",
        "goals",
        "plan",
    ]


def test_timings_are_reported_per_stage():
    result = extract_assessment(TRANSCRIPT, llm=StubLLM())

    timings = result["timings"]

    for stage in (
        "clinicalDetails",
        "objective",
        "grounding",
        "assemble",
        "total",
    ):
        assert stage in timings


# --------------------------------------------------------------------------
# S6 end to end - the headline guarantee
# --------------------------------------------------------------------------


def test_invented_date_never_reaches_the_output():
    """The transcript contains no date, so this one is provably invented."""

    stub = StubLLM(
        overrides={
            "goals": {
                "subjectiveGoals": [
                    {
                        "goalDetails": "improve knee flexion",
                        "targetDate": "2026-12-01",
                    }
                ],
                "objectiveGoals": [],
            }
        }
    )

    result = extract_assessment(TRANSCRIPT, llm=stub)

    dumped = result["assessment"].model_dump()

    assert dumped["subjectiveGoals"][0]["targetDate"] == ""

    assert "2026" not in json.dumps(dumped)

    assert result["confidence"].rejectedCount == 1


def test_invented_measurement_never_reaches_the_output():
    stub = StubLLM(
        overrides={
            "objective": {
                "tests": [
                    {
                        "testName": "Knee flexion",
                        "left": "124",
                        "right": "999",
                    }
                ]
            }
        }
    )

    result = extract_assessment(TRANSCRIPT, llm=stub)

    test = result["assessment"].model_dump()[
        "objectiveAssessment"
    ]["tests"][0]

    assert test["left"] == "124"

    assert test["right"] == ""


def test_rejected_values_are_reported_for_audit():
    stub = StubLLM(
        overrides={
            "plan": {
                "recommendation": [],
                "patientAdvice": {
                    "adviceDetails": "Apply ice three times daily"
                },
            }
        }
    )

    result = extract_assessment(TRANSCRIPT, llm=stub)

    assert result["assessment"].patientAdvice.adviceDetails == ""

    rejected = [
        field
        for field in result["confidence"].flaggedFields
        if field.reason == "rejected"
    ]

    assert any(
        "Apply ice" in flag.detail
        for flag in rejected
    )


# --------------------------------------------------------------------------
# Failure containment
# --------------------------------------------------------------------------


def test_one_failed_section_does_not_lose_the_others():
    """A section that never parses must not take the assessment down."""

    stub = StubLLM(fail={"objective"})

    result = extract_assessment(TRANSCRIPT, llm=stub)

    dumped = result["assessment"].model_dump()

    assert dumped["objectiveAssessment"]["tests"] == []

    assert "objective" in result["errors"]

    # Everything else still arrived.
    assert dumped["clinicalDetails"]["chiefComplaint"] == "left knee pain"

    assert dumped["recommendation"][0]["sessionType"] == "Physiotherapy"


def test_total_failure_still_returns_a_valid_assessment():
    stub = StubLLM(
        fail={
            "clinicalDetails",
            "subjective",
            "objective",
            "goals",
            "plan",
        }
    )

    result = extract_assessment(
        TRANSCRIPT,
        llm=stub,
    )

    assert list(result["assessment"].model_dump()) == list(SECTION_KEYS)

    assert not result["confidence"].meetsThreshold

    assert len(result["errors"]) == 5


def test_llm_transport_failure_propagates():
    """Transport failures are different from structured extraction failures.

    This test intentionally verifies that an unexpected exception from the
    LLM transport is not silently converted into a normal section failure.
    """

    class TransportFailureLLM:
        def invoke(self, messages):
            raise RuntimeError("Ollama connection failed")

    try:
        extract_assessment(
            TRANSCRIPT,
            llm=TransportFailureLLM(),
        )
    except RuntimeError as exc:
        assert "Ollama connection failed" in str(exc)
    else:
        raise AssertionError(
            "Expected LLM transport failure to propagate"
        )