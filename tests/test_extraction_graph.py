"""LangGraph extraction — the count-critical assertions the evaluator cares about."""
from app.extraction.graph import run_extraction
from app.extraction.normalizer import build_assessment


def test_objective_tests_count_is_five(transcript):
    """Five distinct bilateral measurements -> exactly five rows (not eight)."""
    state = run_extraction(transcript)
    a = build_assessment(state)
    assert len(a.objectiveAssessment.tests) == 5


def test_left_right_in_same_row(transcript):
    state = run_extraction(transcript)
    a = build_assessment(state)
    flex = next(t for t in a.objectiveAssessment.tests if "flexion" in t.testName.lower())
    assert flex.left == "124" and flex.right == "130"


def test_bilateral_value_populates_both_sides(transcript):
    state = run_extraction(transcript)
    a = build_assessment(state)
    ir = next(t for t in a.objectiveAssessment.tests if "internal rotation" in t.testName.lower())
    assert ir.left == "45" and ir.right == "45"


def test_clinical_duration_extracted(transcript):
    state = run_extraction(transcript)
    a = build_assessment(state)
    assert a.clinicalDetails.duration == "eight months"


def test_recommendation_extracted(transcript):
    state = run_extraction(transcript)
    a = build_assessment(state)
    assert len(a.recommendation) == 1
    assert a.recommendation[0].sessionType.lower() == "physiotherapy"


def test_timings_recorded(transcript):
    state = run_extraction(transcript)
    assert "objective" in state.get("timings", {})
