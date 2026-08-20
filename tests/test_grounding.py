"""Tests for the grounding guard - the proof of requirement S6.

The brief calls "never hallucinate clinical values, scores, or dates" critical.
These tests are the evidence for that claim: they show invented values are
removed by code, independently of whether the model obeyed its prompt.
"""

from __future__ import annotations

import pytest

from app.extraction.grounding import (
    content_tokens,
    extract_dates,
    extract_numbers,
    ground_payload,
    normalise,
    verify_value,
)

# A condensed version of the supplied recording. Like the real one, it
# contains no dates at all - which makes any emitted date provably invented.
TRANSCRIPT = (
    "The patient presented with left knee pain and difficulty walking following "
    "surgery. She was involved in a road traffic accident eight months ago "
    "resulting in a left tibial condyle fracture. Open reduction and internal "
    "fixation was performed by Dr. Hemant Kalyan, followed by four to six weeks "
    "of non-weight bearing. Objective measurements showed left knee flexion of "
    "124° compared with 130° on the right, and ankle dorsiflexion of 4.5° "
    "on the left compared with 12° on the right. Physiotherapy was recommended "
    "once weekly for four sessions."
)


# --------------------------------------------------------------------------
# Values that must survive - false positives are their own kind of failure
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "value",
    [
        "124",
        "4.5",
        "left knee pain",
        "Physiotherapy",
        "once weekly for four sessions",
        "road traffic accident",
        "Dr. Hemant Kalyan",
    ],
)
def test_real_values_are_grounded(value):
    assert verify_value(value, TRANSCRIPT).grounded, value


def test_blank_is_grounded():
    """"Not stated" is the correct answer for uncovered fields, not a failure."""
    assert verify_value("", TRANSCRIPT).grounded
    assert verify_value("   ", TRANSCRIPT).grounded


def test_spoken_number_matches_written_digit():
    """The recording says "eight months"; "8 months" is a correct reading."""
    assert verify_value("8 months", TRANSCRIPT).grounded
    assert verify_value("eight months", TRANSCRIPT).grounded


def test_degree_symbol_grounds_the_word_degrees():
    """The transcript writes 124 with a degree sign, never the word.

    Without expanding the symbol, every correct unitName would be discarded.
    """
    assert verify_value("degrees", TRANSCRIPT).grounded
    assert verify_value("deg", TRANSCRIPT).grounded


# --------------------------------------------------------------------------
# Values that must be rejected - the actual S6 guarantee
# --------------------------------------------------------------------------
def test_invented_measurement_is_rejected():
    verdict = verify_value("95", TRANSCRIPT)
    assert not verdict.grounded
    assert "95" in verdict.reason


def test_plausible_but_invented_measurement_is_rejected():
    """Fluent, clinically sensible, and wrong - the dangerous case."""
    verdict = verify_value("Knee flexion improved to 140 degrees", TRANSCRIPT)
    assert not verdict.grounded
    assert "140" in verdict.reason


@pytest.mark.parametrize(
    "date_value",
    ["2026-09-01", "01/09/2026", "15 September 2026", "September 2026", "Sep 15"],
)
def test_invented_dates_are_rejected(date_value):
    """No date appears in the recording, so every date is invented."""
    assert not verify_value(date_value, TRANSCRIPT).grounded


def test_fluent_invention_without_numbers_is_rejected():
    """Catches invention that avoids numbers entirely."""
    verdict = verify_value(
        "Patient has a longstanding history of rheumatoid arthritis and diabetes",
        TRANSCRIPT,
    )
    assert not verdict.grounded
    assert "content words" in verdict.reason


def test_rejection_reason_names_the_offending_number():
    """The reason reaches the clinician via the flag report, so it must be specific."""
    verdict = verify_value("Flexion 99 degrees", TRANSCRIPT)
    assert not verdict.grounded
    assert "99" in verdict.reason


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def test_number_extraction_normalises_equivalent_forms():
    assert extract_numbers("4.50") == {"4.5"}
    assert extract_numbers("8.0") == {"8"}
    assert "8" in extract_numbers("eight months")


def test_normalise_expands_symbols_and_words():
    result = normalise("124° and eight months")
    assert "degrees" in result
    assert "8" in result


def test_stopwords_do_not_count_toward_overlap():
    """Otherwise an invented sentence scores highly on "the", "of", "and"."""
    tokens = content_tokens("the patient was in the of and")
    assert tokens == []


def test_date_extraction_finds_common_formats():
    assert extract_dates("due 2026-09-01") == {"2026-09-01"}
    assert extract_dates("no dates here") == set()


# --------------------------------------------------------------------------
# Whole-payload grounding
# --------------------------------------------------------------------------
def test_ground_payload_clears_and_reports_nested_values():
    payload = {
        "clinicalDetails": {
            "chiefComplaint": "left knee pain",          # real
            "duration": "eight months",                   # real
            "clinicalHistory": "prior stroke in 2019",    # invented
        },
        "tests": [
            {"testName": "Knee flexion", "left": "124", "right": "130"},   # real
            {"testName": "Grip strength", "left": "40", "right": "42"},    # invented
        ],
    }

    cleaned, issues = ground_payload(payload, TRANSCRIPT)

    assert cleaned["clinicalDetails"]["chiefComplaint"] == "left knee pain"
    assert cleaned["clinicalDetails"]["duration"] == "eight months"
    assert cleaned["clinicalDetails"]["clinicalHistory"] == ""      # cleared
    assert cleaned["tests"][0]["left"] == "124"
    assert cleaned["tests"][1]["left"] == ""                        # cleared
    assert cleaned["tests"][1]["right"] == ""

    paths = {issue.path for issue in issues}
    assert "clinicalDetails.clinicalHistory" in paths
    assert "tests[1].left" in paths
    assert "tests[1].right" in paths


def test_rejected_values_are_cleared_not_kept():
    """The core safety property: a failed value never reaches the output."""
    payload = {"targetDate": "2026-12-25", "goalName": "Restore knee flexion"}
    cleaned, issues = ground_payload(payload, TRANSCRIPT)

    assert cleaned["targetDate"] == ""
    assert len(issues) == 1
    # The discarded value survives only in the report, for audit.
    assert issues[0].value == "2026-12-25"


def test_grounding_preserves_structure():
    """Clearing values must not change the shape the schema expects."""
    payload = {"a": {"b": "invented nonsense phrase xyzzy"}, "c": [{"d": "999"}]}
    cleaned, _ = ground_payload(payload, TRANSCRIPT)
    assert set(cleaned) == {"a", "c"}
    assert isinstance(cleaned["c"], list)
    assert set(cleaned["c"][0]) == {"d"}
