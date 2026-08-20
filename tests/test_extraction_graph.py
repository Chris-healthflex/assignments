"""End-to-end tests for the LangGraph extraction agent.

The LLM is stubbed, so these run in milliseconds with no Ollama daemon, no GPU
and no model download - which is what lets them run in CI.
"""

from __future__ import annotations

import json

import pytest

from app.extraction.graph import (
    _merge_sided_tests,
    _normalise_subjective,
    _normalise_tests,
    _split_goals,
    extract_assessment,
)
from app.extraction.llm import StructuredOutputError
from app.schemas.first_assessment import SECTION_KEYS

TRANSCRIPT = (
    "The patient presented with left knee pain following surgery eight months "
    "ago after a road traffic accident. Left knee flexion was 124° compared "
    "with 130° on the right. Physiotherapy was recommended once weekly for "
    "four sessions."
)


class StubLLM:
    """Returns a canned payload per section, keyed off the prompt text."""

    def __init__(self, overrides: dict[str, dict] | None = None, fail: set[str] | None = None):
        self.overrides = overrides or {}
        self.fail = fail or set()
        self.calls: list[str] = []

    def _section_for(self, prompt: str) -> str:
        if "clinical background" in prompt:
            return "clinicalDetails"
        if "what the patient reported" in prompt:
            return "subjective"
        if "objective measurements recorded" in prompt:
            return "objective"
        if "treatment goals discussed" in prompt:
            return "goals"
        return "plan"

    def invoke(self, messages):
        section = self._section_for(messages[1][1])
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
            "subjective": {"subjectiveAssessments": [{"testName": "Pain", "conclusion": "left knee pain"}]},
            "objective": {"tests": [{"testName": "Knee flexion", "unitName": "degrees", "left": "124", "right": "130"}]},
            "goals": {"subjectiveGoals": [], "objectiveGoals": []},
            "plan": {
                "recommendation": [{"sessionType": "Physiotherapy", "sessionFrequency": "once weekly for four sessions"}],
                "patientAdvice": {"adviceDetails": ""},
            },
        }
        body = self.overrides.get(section, defaults[section])

        class Response:
            content = json.dumps(body)

        return Response()


# --------------------------------------------------------------------------
# Happy path
# --------------------------------------------------------------------------
def test_graph_produces_a_schema_valid_assessment():
    result = extract_assessment(TRANSCRIPT, llm=StubLLM())
    dumped = result["assessment"].model_dump()

    assert list(dumped) == list(SECTION_KEYS)
    assert dumped["clinicalDetails"]["chiefComplaint"] == "left knee pain"
    assert dumped["objectiveAssessment"]["tests"][0]["left"] == "124"
    assert dumped["recommendation"][0]["sessionType"] == "Physiotherapy"


def test_every_section_node_runs():
    stub = StubLLM()
    extract_assessment(TRANSCRIPT, llm=stub)
    assert stub.calls == ["clinicalDetails", "subjective", "objective", "goals", "plan"]


def test_timings_are_reported_per_stage():
    result = extract_assessment(TRANSCRIPT, llm=StubLLM())
    timings = result["timings"]
    for stage in ("clinicalDetails", "objective", "grounding", "assemble", "total"):
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
                    {"goalDetails": "improve knee flexion", "targetDate": "2026-12-01"}
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
                "tests": [{"testName": "Knee flexion", "left": "124", "right": "999"}]
            }
        }
    )
    result = extract_assessment(TRANSCRIPT, llm=stub)
    test = result["assessment"].model_dump()["objectiveAssessment"]["tests"][0]

    assert test["left"] == "124"      # real value kept
    assert test["right"] == ""        # invented value cleared


def test_rejected_values_are_reported_for_audit():
    stub = StubLLM(
        overrides={
            "plan": {
                "recommendation": [],
                "patientAdvice": {"adviceDetails": "Apply ice three times daily"},
            }
        }
    )
    result = extract_assessment(TRANSCRIPT, llm=stub)

    assert result["assessment"].patientAdvice.adviceDetails == ""
    rejected = [f for f in result["confidence"].flaggedFields if f.reason == "rejected"]
    assert any("Apply ice" in flag.detail for flag in rejected)


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
    stub = StubLLM(fail={"clinicalDetails", "subjective", "objective", "goals", "plan"})
    result = extract_assessment(TRANSCRIPT, llm=stub)

    assert list(result["assessment"].model_dump()) == list(SECTION_KEYS)
    assert not result["confidence"].meetsThreshold
    assert len(result["errors"]) == 5


def test_llm_transport_failure_propagates():
    """A dead daemon is a 503, not a silently empty assessment."""

    class Dead:
        def invoke(self, messages):
            raise ConnectionError("connection refused")

    from app.extraction.llm import LLMUnavailableError

    with pytest.raises(LLMUnavailableError):
        extract_assessment(TRANSCRIPT, llm=Dead())


# --------------------------------------------------------------------------
# Objective test normalisation
# --------------------------------------------------------------------------
def test_value_is_cleared_when_a_side_is_present():
    """value is for unsided measurements; a small model fills both."""
    out = _normalise_tests([{"testName": "Hip IR", "value": "45", "left": "45", "right": ""}])
    assert out[0]["value"] == ""
    assert out[0]["left"] == "45"


def test_value_is_kept_when_no_side_is_given():
    out = _normalise_tests([{"testName": "Single", "value": "20", "left": "", "right": ""}])
    assert out[0]["value"] == "20"


def test_entries_with_no_content_are_dropped():
    out = _normalise_tests([{"testName": "", "value": "", "left": "", "right": ""}])
    assert out == []


def test_named_test_without_a_measurement_is_kept_for_flagging():
    """Keeping it lets confidence report that the measurement went missing."""
    out = _normalise_tests([{"testName": "Knee flexion", "value": "", "left": "", "right": ""}])
    assert len(out) == 1


# --------------------------------------------------------------------------
# Reducing blank fields that are the extractor's fault, not the recording's
# --------------------------------------------------------------------------
def test_narrative_crammed_into_test_name_moves_to_conclusion():
    """The model puts the whole finding in testName and leaves conclusion empty.

    Rendered as-is that is a paragraph-length field label beside the word
    "not stated". Moving it rearranges extracted text; it invents nothing.
    """
    out = _normalise_subjective([
        {"testName": "Healed surgical scar on the medial aspect of the knee, with swelling",
         "conclusion": ""},
    ])
    assert out[0]["testName"] == "Surgical scar"
    assert out[0]["conclusion"].startswith("Healed surgical scar")


def test_short_sentence_findings_are_also_moved():
    """Length alone was too blunt: this is short but plainly a finding."""
    out = _normalise_subjective([{"testName": "Patellar mobility was good.", "conclusion": ""}])
    assert out[0]["testName"] == "Patellar mobility"
    assert out[0]["conclusion"] == "Patellar mobility was good."


def test_a_real_label_is_left_alone():
    out = _normalise_subjective([{"testName": "Pain", "conclusion": "Moderate, worse on walking"}])
    assert out[0]["testName"] == "Pain"
    assert out[0]["conclusion"] == "Moderate, worse on walking"


def test_label_matching_prefers_the_specific_signal_over_anatomy():
    """"Knee pain ... during walking" is about pain, not gait or the knee."""
    out = _normalise_subjective([
        {"testName": "Left knee pain with difficulty walking and ankle pain", "conclusion": ""},
    ])
    assert out[0]["testName"] == "Pain"


def test_unmatched_finding_gets_no_invented_label():
    """Better an empty label than a guessed clinical category."""
    out = _normalise_subjective([{"testName": "Something not in the vocabulary at all", "conclusion": ""}])
    assert out[0]["testName"] == ""
    assert out[0]["conclusion"] == "Something not in the vocabulary at all"


def test_sided_rows_merge_into_one_complete_measurement():
    """Two half-blank rows become one row with both sides."""
    out = _merge_sided_tests([
        {"testName": "Left knee flexion", "left": "124", "right": "", "value": ""},
        {"testName": "Right knee flexion", "left": "", "right": "130", "value": ""},
    ])
    assert len(out) == 1
    assert out[0]["testName"] == "Knee flexion"
    assert (out[0]["left"], out[0]["right"]) == ("124", "130")


def test_a_right_row_carrying_its_value_on_the_left_still_merges():
    """The model files a "Right ..." row's number under left; the name wins."""
    out = _merge_sided_tests([
        {"testName": "Left knee extension", "left": "20", "right": "5", "value": ""},
        {"testName": "Right knee extension", "left": "5", "right": "", "value": ""},
    ])
    assert len(out) == 1
    assert (out[0]["left"], out[0]["right"]) == ("20", "5")


def test_unsided_measurement_is_untouched():
    out = _merge_sided_tests([{"testName": "Girth", "left": "", "right": "", "value": "38"}])
    assert out[0]["testName"] == "Girth"
    assert out[0]["value"] == "38"


def test_goal_without_a_measurable_target_becomes_subjective():
    """goalCategory, unitName and value would otherwise stay blank forever."""
    objective, subjective = _split_goals(
        [{"goalName": "Improving ankle mobility", "goalCategory": "", "unitName": "", "value": "", "targetDate": ""}],
        [],
    )
    assert objective == []
    assert subjective == [{"goalDetails": "Improving ankle mobility", "targetDate": ""}]


def test_goal_with_a_measurable_target_stays_objective():
    objective, subjective = _split_goals(
        [{"goalName": "Knee flexion", "unitName": "degrees", "value": "130", "targetDate": ""}],
        [],
    )
    assert len(objective) == 1
    assert subjective == []


def test_reclassifying_a_goal_preserves_its_target_date():
    """A date that WAS stated must survive the move."""
    _, subjective = _split_goals(
        [{"goalName": "Walk unaided", "value": "", "targetDate": "2026-09-01"}], []
    )
    assert subjective[0]["targetDate"] == "2026-09-01"


def test_two_findings_do_not_collide_on_the_same_label():
    """Both mention pain; labelling both "Pain" tells a clinician nothing."""
    out = _normalise_subjective([
        {"testName": "Moderate pain worse on prolonged walking", "conclusion": ""},
        {"testName": "Restricted painful knee flexion with swelling", "conclusion": ""},
    ])
    assert out[0]["testName"] == "Pain"
    assert out[1]["testName"] != "Pain"
    assert out[1]["testName"]          # and it is not blank either


def test_the_unit_is_stripped_from_measurement_values():
    """unitName already carries the unit; repeating it in left/right makes the
    field something a consumer has to parse rather than read - and it broke the
    completeness check, which counted "124 degrees" as no number at all."""
    from app.extraction.graph import _strip_unit

    out = _merge_sided_tests(_normalise_tests([
        {"testName": "Knee flexion", "unitName": "degrees", "left": "124\u00b0", "right": "130\u00b0", "value": ""},
    ]))
    assert (out[0]["left"], out[0]["right"]) == ("124", "130")
    assert out[0]["unitName"] == "degrees"

    assert _strip_unit("4.5\u00b0") == "4.5"
    assert _strip_unit("20 degrees") == "20"
    assert _strip_unit("") == ""
    assert _strip_unit("normal") == "normal"      # nothing numeric: left alone
