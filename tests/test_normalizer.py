"""Normalizer maps raw state -> validated FirstAssessment, dropping stray keys."""
from app.extraction.normalizer import build_assessment


def test_unknown_keys_dropped():
    state = {
        "clinicalDetails": {"duration": "eight months", "bogus": "x"},
        "objectiveTests": [{"testName": "Knee flexion", "left": "124", "junk": 1}],
    }
    a = build_assessment(state)  # must not raise despite extra keys
    assert a.clinicalDetails.duration == "eight months"
    assert a.objectiveAssessment.tests[0].left == "124"


def test_missing_sections_default_empty():
    a = build_assessment({})
    assert a.subjectiveAssessments == []
    assert a.patientAdvice.adviceDetails == ""
