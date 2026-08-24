"""End-to-end pipeline from a transcript."""
from app.services.pipeline import run_from_transcript


def test_pipeline_envelope(transcript):
    result = run_from_transcript(transcript)
    d = result.model_dump(mode="json")
    for key in ("assessment", "transcript", "confidence", "flaggedFields", "timings"):
        assert key in d
    assert len(d["assessment"]["objectiveAssessment"]["tests"]) == 5
    assert d["confidence"]["overall"] >= d["confidence"]["threshold"]


def test_no_hallucinated_targetdates(transcript):
    result = run_from_transcript(transcript)
    for g in result.assessment.objectiveGoals:
        assert g.targetDate == ""  # transcript states none
