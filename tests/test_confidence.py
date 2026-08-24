"""Confidence scoring + flagging."""
from app.schemas.assessment import FirstAssessment, ClinicalDetails
from app.extraction.grounding import GroundingReport
from app.extraction.confidence import score


def test_empty_scalars_flagged_not_stated():
    a = FirstAssessment()
    res = score(a, GroundingReport(), threshold=0.55)
    paths = {f["path"] for f in res.flagged}
    assert "patientAdvice.adviceDetails" in paths
    assert "subjectiveGoals" in paths
    assert all(f["reason"] in {"not_stated", "ungrounded"} for f in res.flagged)


def test_overall_is_mean_of_sections():
    a = FirstAssessment(clinicalDetails=ClinicalDetails(
        clinicalHistory="h", chiefComplaint="c", duration="d"))
    res = score(a, GroundingReport(), threshold=0.55)
    assert res.section_scores["clinicalDetails"] == 1.0
    assert 0.0 <= res.overall <= 1.0


def test_threshold_flag():
    a = FirstAssessment()  # everything empty -> low score
    res = score(a, GroundingReport(), threshold=0.55)
    assert res.meets_threshold is False


def test_rejected_count_matches_ungrounded():
    g = GroundingReport()
    g.flag("objectiveAssessment.tests[0].left", "x")
    res = score(FirstAssessment(), g, threshold=0.55)
    assert res.rejected_count == 1
