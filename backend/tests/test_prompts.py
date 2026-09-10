"""Tests for the extraction and repair prompt builders."""

import pytest

from clinical_assessment.prompts import (
    EXTRACTION_SYSTEM,
    build_extraction_prompt,
    build_repair_prompt,
)

TRANSCRIPT = (
    "Clinician: What brings you in today? Patient: My right shoulder has been "
    "painful for about three weeks. Clinician: Abduction measures 120 degrees "
    "on the right."
)


def test_prompt_embeds_the_full_transcript_untruncated() -> None:
    long_transcript = TRANSCRIPT * 200

    prompt = build_extraction_prompt(long_transcript)

    assert long_transcript in prompt


def test_extraction_system_states_the_verbatim_evidence_contract() -> None:
    assert "VERBATIM" in EXTRACTION_SYSTEM


def test_repair_prompt_names_only_the_ungrounded_fields() -> None:
    rejected_quotes = {
        "objectiveAssessment.tests[0].value": "flexion was 45 degrees",
        "clinicalDetails.duration": "for about six months",
    }

    prompt = build_repair_prompt(TRANSCRIPT, rejected_quotes)

    assert [line for line in prompt.splitlines() if line.startswith("- ")] == [
        "- objectiveAssessment.tests[0].value: quoted 'flexion was 45 degrees'",
        "- clinicalDetails.duration: quoted 'for about six months'",
    ]


def test_repair_prompt_embeds_the_transcript_for_rereading() -> None:
    prompt = build_repair_prompt(TRANSCRIPT, {"clinicalDetails.duration": "six months"})

    assert TRANSCRIPT in prompt


def test_repair_prompt_raises_when_no_fields_failed() -> None:
    with pytest.raises(ValueError, match="at least one failed field"):
        build_repair_prompt(TRANSCRIPT, {})
