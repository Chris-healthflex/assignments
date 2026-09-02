"""
Offline tests (no Whisper / LLM / Mongo needed):  pytest tests/test_schema.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.agent import audit  # noqa: E402
from app.schemas import ExtractionDraft, FirstAssessment  # noqa: E402

EXPECTED_KEYS = {
    "clinicalDetails", "subjectiveAssessments", "objectiveAssessment",
    "subjectiveGoals", "objectiveGoals", "recommendation", "patientAdvice",
}


def test_empty_assessment_has_exact_top_level_keys():
    assert set(FirstAssessment().model_dump()) == EXPECTED_KEYS


def test_nested_keys_match_brief():
    d = FirstAssessment().model_dump()
    assert set(d["clinicalDetails"]) == {"clinicalHistory", "chiefComplaint", "duration"}
    assert d["objectiveAssessment"] == {"tests": []}
    assert d["patientAdvice"] == {"adviceDetails": ""}


def test_extra_keys_rejected():
    with pytest.raises(ValidationError):
        FirstAssessment.model_validate({"clinicalDetails": {}, "diagnosis": "x"})


def test_null_strings_become_empty_and_lists_stay_lists():
    a = FirstAssessment.model_validate({
        "clinicalDetails": {"clinicalHistory": None, "chiefComplaint": "knee pain", "duration": None},
        "recommendation": [{"sessionType": "Physio", "sessionFrequency": None}],
    })
    assert a.clinicalDetails.clinicalHistory == ""
    assert a.recommendation[0].sessionFrequency == ""
    assert isinstance(a.subjectiveGoals, list)


def test_null_list_rejected():
    with pytest.raises(ValidationError):
        FirstAssessment.model_validate({"subjectiveGoals": None})


def test_audit_flags_hallucinated_numbers_and_missing_core_fields():
    transcript = "Patient has right knee pain for two weeks. Flexion measured at 110 degrees."
    draft = ExtractionDraft(
        assessment=FirstAssessment.model_validate({
            "clinicalDetails": {"chiefComplaint": "right knee pain", "duration": "two weeks"},
            "objectiveAssessment": {"tests": [
                {"testName": "Knee flexion", "unitName": "degrees", "right": "110"},
                {"testName": "Knee extension", "unitName": "degrees", "right": "5"},  # 5 never said
            ]},
        }),
        overall_confidence=0.9,
    )
    out = audit({"transcript": transcript, "assessment": draft.assessment, "flags": [],
                 "overall_confidence": 0.9, "session_date": None})
    paths = {f.field for f in out["flags"]}
    assert "objectiveAssessment.tests[1].right" in paths     # hallucinated 5
    assert "objectiveAssessment.tests[0].right" not in paths  # 110 is in transcript
    assert "clinicalDetails.clinicalHistory" in paths         # empty core field
    assert out["low_confidence"] is True                      # core field missing -> 422
