from __future__ import annotations

import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.agent import audit, run_extraction_pipeline
from app.schemas import ExtractionDraft, FirstAssessment, ObjectiveAssessment, ObjectiveTest

EXPECTED_KEYS = {
    "clinicalDetails",
    "subjectiveAssessments",
    "objectiveAssessment",
    "subjectiveGoals",
    "objectiveGoals",
    "recommendation",
    "patientAdvice",
}


def test_empty_assessment_has_exact_top_level_keys():
    dump = FirstAssessment().model_dump()
    assert set(dump.keys()) == EXPECTED_KEYS


def test_nested_keys_match_brief():
    d = FirstAssessment().model_dump()
    assert set(d["clinicalDetails"].keys()) == {"clinicalHistory", "chiefComplaint", "duration"}
    assert d["objectiveAssessment"] == {"tests": []}
    assert d["patientAdvice"] == {"adviceDetails": ""}
    assert d["subjectiveAssessments"] == []
    assert d["subjectiveGoals"] == []
    assert d["objectiveGoals"] == []
    assert d["recommendation"] == []


def test_extra_keys_rejected():
    with pytest.raises(ValidationError):
        FirstAssessment.model_validate({"clinicalDetails": {}, "unexpectedDiagnosisKey": "foo"})


def test_null_strings_become_empty_and_lists_stay_lists():
    a = FirstAssessment.model_validate({
        "clinicalDetails": {
            "clinicalHistory": None,
            "chiefComplaint": "Knee pain",
            "duration": None,
        },
        "recommendation": [{"sessionType": "Physio", "sessionFrequency": None}],
        "patientAdvice": {"adviceDetails": None},
    })
    assert a.clinicalDetails.clinicalHistory == ""
    assert a.clinicalDetails.duration == ""
    assert a.recommendation[0].sessionFrequency == ""
    assert a.patientAdvice.adviceDetails == ""
    assert isinstance(a.subjectiveGoals, list)
    assert isinstance(a.objectiveAssessment.tests, list)


def test_null_list_rejected():
    with pytest.raises(ValidationError):
        FirstAssessment.model_validate({"subjectiveGoals": None})


def test_audit_flags_hallucinated_numbers_and_missing_core_fields():
    transcript = "Patient has right knee pain for two weeks. Flexion measured at 110 degrees."
    draft = ExtractionDraft(
        assessment=FirstAssessment.model_validate({
            "clinicalDetails": {"chiefComplaint": "Right knee pain", "duration": "two weeks"},
            "objectiveAssessment": {
                "tests": [
                    {"testName": "Knee flexion", "unitName": "degrees", "right": "110"},
                    {"testName": "Knee extension", "unitName": "degrees", "right": "52"},  # 52 never said in transcript
                ]
            },
        }),
        overall_confidence=0.9,
    )
    out = audit({
        "transcript": transcript,
        "assessment": draft.assessment,
        "flags": [],
        "overall_confidence": 0.9,
        "session_date": None,
        "raw_draft": None,
        "low_confidence": False,
    })
    paths = {f.field for f in out["flags"]}
    assert "objectiveAssessment.tests[1].right" in paths     # Hallucinated 52
    assert "objectiveAssessment.tests[0].right" not in paths  # 110 is in transcript
    assert "clinicalDetails.clinicalHistory" in paths         # Missing core field
    assert out["low_confidence"] is True                      # Triggers low_confidence


def test_extraction_pipeline_on_full_clinical_transcript():
    transcript = (
        "The patient presented with left knee pain and difficulty walking. "
        "Road traffic accident eight months ago resulting in tibial condylar fracture. "
        "Knee flexion of 124 degrees on the left compared to 130 on the right. "
        "Physiotherapy recommended once weekly for 4 sessions."
    )
    res = run_extraction_pipeline(transcript)
    assert res.overall_confidence >= 0.70
    assert not res.low_confidence
    assert res.assessment.clinicalDetails.chiefComplaint != ""
    assert len(res.assessment.objectiveAssessment.tests) > 0
