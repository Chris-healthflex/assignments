import pytest
from app.services.grounding import (
    ground_number,
    ground_date_or_duration,
    ground_field,
    extract_numbers,
    number_to_spoken_words,
)
from app.services.confidence import fuse_confidence, evaluate_confidence
from app.models.internal import FieldEvidence


def test_spoken_words_generation():
    forms_52 = number_to_spoken_words(52)
    assert "52" in forms_52
    assert "fifty two" in forms_52 or "fifty-two" in forms_52

    forms_10 = number_to_spoken_words(10)
    assert "10" in forms_10
    assert "ten" in forms_10


def test_extract_numbers():
    assert extract_numbers("52 degrees") == [52]
    assert extract_numbers("pain was 4/10 today") == [4, 10]
    assert extract_numbers("no numbers here") == []


def test_ground_number_digits_and_spoken():
    transcript = "Patient reports left knee flexion reached 110 degrees, with right knee at fifty two degrees."

    # Digits match
    grounded, span = ground_number(transcript, 110, anchor="flexion")
    assert grounded is True
    assert "110" in span

    # Spoken words match
    grounded, span = ground_number(transcript, 52, anchor="right knee")
    assert grounded is True
    assert "fifty two" in span.lower()

    # Ungrounded number (not present anywhere in transcript)
    grounded, span = ground_number(transcript, 140, anchor="flexion")
    assert grounded is False
    assert span is None


def test_ground_date_or_duration():
    transcript = "Symptoms started approximately six weeks ago after a football match."

    # Number word match
    grounded, span = ground_date_or_duration(transcript, "6 weeks")
    assert grounded is True
    assert "six weeks" in span.lower()

    # Ungrounded duration
    grounded, span = ground_date_or_duration(transcript, "3 months")
    assert grounded is False
    assert span is None


def test_ground_field_dispatch():
    transcript = "On examination, active left knee extension was 0 degrees and pain was 4 out of 10."

    # Numeric field
    grounded, span = ground_field(transcript, "objectiveAssessment.tests[0].value", "0 degrees", anchor="extension")
    assert grounded is True
    assert span is not None

    # Ungrounded numeric field
    grounded, span = ground_field(transcript, "objectiveAssessment.tests[1].value", "120 degrees", anchor="flexion")
    assert grounded is False
    assert span is None

    # Empty value should not fail
    grounded, span = ground_field(transcript, "objectiveAssessment.tests[2].value", "")
    assert grounded is True
    assert span is None


def test_fusion_rule_hard_cap():
    # Overconfident hallucinated number: LLM claims 0.95, transcript does not support it
    fused = fuse_confidence(llm_confidence=0.95, grounded=False, is_numeric_or_date=True)
    assert fused == 0.35, "Ungrounded numeric field MUST be hard-capped at <= 0.35"

    # Confident grounded number: LLM claims 0.95, transcript supports it
    fused = fuse_confidence(llm_confidence=0.95, grounded=True, is_numeric_or_date=True)
    assert fused == 0.95

    # Low confidence ungrounded number: LLM claims 0.20 -> stays 0.20
    fused = fuse_confidence(llm_confidence=0.20, grounded=False, is_numeric_or_date=True)
    assert fused == 0.20

    # Non-numeric ungrounded field: not subject to hard numeric cap
    fused = fuse_confidence(llm_confidence=0.85, grounded=False, is_numeric_or_date=False)
    assert fused == 0.85


def test_evaluate_confidence_overall_minimum():
    evidence_items = [
        FieldEvidence(
            field="clinicalDetails.chiefComplaint",
            confidence=0.90,
            llm_confidence=0.90,
            grounded=True,
            evidence_span="left knee pain",
            reason="Directly stated",
        ),
        FieldEvidence(
            field="objectiveAssessment.tests[0].value",
            confidence=0.35,  # Capped because ungrounded
            llm_confidence=0.95,
            grounded=False,
            evidence_span=None,
            reason="Numeric value 140 not found in transcript",
        ),
        FieldEvidence(
            field="patientAdvice.adviceDetails",
            confidence=0.88,
            llm_confidence=0.88,
            grounded=True,
            evidence_span="ice and rest",
            reason="Directly stated",
        ),
    ]

    report = evaluate_confidence(evidence_items, threshold=0.70)
    # Overall is min(0.90, 0.35, 0.88) = 0.35
    assert report.overall == 0.35
    assert report.passed is False
    assert len(report.flags) == 1
    assert report.flags[0].field == "objectiveAssessment.tests[0].value"
