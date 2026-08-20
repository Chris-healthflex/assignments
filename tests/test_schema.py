"""Schema contract tests -- these must pass without any external service."""

import pytest
from pydantic import ValidationError

from app.schemas import (
    Complaint,
    ExtractionPayload,
    FirstAssessment,
    PainQuality,
    Patient,
    Sex,
)


def test_empty_assessment_is_valid():
    """Every clinical field is optional: a silent transcript must not crash us."""
    assessment = FirstAssessment()
    assert assessment.complaints == []
    assert assessment.meta.schema_version == "0.1.0"
    assert assessment.created_at is not None


def test_full_assessment_round_trips():
    assessment = FirstAssessment(
        patient=Patient(name="Jane Doe", age=41, sex=Sex.female),
        complaints=[
            Complaint(
                body_region="lower back",
                side="left",
                pain_score=6,
                quality=[PainQuality.aching, PainQuality.stiff],
                duration="about three weeks",
                is_primary=True,
            )
        ],
    )
    restored = FirstAssessment.model_validate(assessment.model_dump())
    assert restored.patient.name == "Jane Doe"
    assert restored.complaints[0].pain_score == 6


@pytest.mark.parametrize("score", [-1, 11])
def test_pain_score_out_of_range_rejected(score):
    with pytest.raises(ValidationError):
        Complaint(pain_score=score)


def test_unknown_field_rejected():
    """extra='forbid' means schema drift fails loudly rather than silently."""
    with pytest.raises(ValidationError):
        Patient(name="Jane", nickname="Janey")


def test_extraction_payload_defaults_are_empty():
    payload = ExtractionPayload()
    assert payload.unresolved_fields == []
    assert payload.medical_history.red_flags == []
