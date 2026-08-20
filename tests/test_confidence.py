"""Tests for confidence scoring and field flagging (S5).

The distinction these lock down: a field that was never discussed
("not_stated") is benign, while a value that failed grounding ("rejected") is
a caught hallucination. Reporting them identically would hide the difference
that matters to a reviewing clinician.
"""

from __future__ import annotations

from app.extraction.confidence import SECTION_WEIGHTS, score
from app.extraction.grounding import GroundingIssue
from app.schemas.first_assessment import FirstAssessment, empty_assessment

THRESHOLD = 0.55

FULL_PAYLOAD = {
    "clinicalDetails": {
        "clinicalHistory": "road traffic accident with tibial condyle fracture",
        "chiefComplaint": "left knee pain",
        "duration": "eight months",
    },
    "subjectiveAssessments": [{"testName": "Pain", "conclusion": "moderate"}],
    "objectiveAssessment": {
        "tests": [{"testName": "Knee flexion", "left": "124", "right": "130"}]
    },
    "subjectiveGoals": [{"goalDetails": "walk unaided"}],
    "objectiveGoals": [{"goalName": "Restore extension"}],
    "recommendation": [{"sessionType": "Physiotherapy", "sessionFrequency": "weekly"}],
    "patientAdvice": {"adviceDetails": "home exercises daily"},
}


def test_section_weights_sum_to_one():
    assert sum(SECTION_WEIGHTS.values()) == 1.0


def test_empty_assessment_scores_zero_and_fails_threshold():
    report = score(empty_assessment(), [], threshold=THRESHOLD)
    assert report.overall == 0.0
    assert not report.meetsThreshold


def test_complete_assessment_scores_one():
    assessment = FirstAssessment.model_validate(FULL_PAYLOAD)
    report = score(assessment, [], threshold=THRESHOLD)
    assert report.overall == 1.0
    assert report.meetsThreshold
    assert report.rejectedCount == 0


def test_score_and_flags_are_independent_axes():
    """A section can score 1.0 and still contain fields worth checking.

    The goals above carry no targetDate, which does not reduce completeness of
    the core record but is still worth putting in front of a clinician. Score
    answers "is this assessment usable"; flags answer "what should I check".
    """
    assessment = FirstAssessment.model_validate(FULL_PAYLOAD)
    report = score(assessment, [], threshold=THRESHOLD)

    assert report.overall == 1.0
    date_flags = [f for f in report.flaggedFields if f.path.endswith("targetDate")]
    assert date_flags
    assert all(flag.reason == "not_stated" for flag in date_flags)


def test_clinical_details_scores_proportionally():
    assessment = FirstAssessment.model_validate(
        {"clinicalDetails": {"chiefComplaint": "knee pain"}}
    )
    report = score(assessment, [], threshold=THRESHOLD)
    assert report.sectionScores["clinicalDetails"] == round(1 / 3, 3)


def test_missing_core_sections_drag_the_score_below_threshold():
    """A recording with only patient advice is not a usable assessment."""
    assessment = FirstAssessment.model_validate(
        {"patientAdvice": {"adviceDetails": "rest"}}
    )
    report = score(assessment, [], threshold=THRESHOLD)
    assert report.overall < THRESHOLD
    assert not report.meetsThreshold


def test_optional_sections_do_not_sink_an_otherwise_good_assessment():
    """No goals and no advice is normal, and must still pass the gate.

    Otherwise the supplied recording - which discusses neither - would 422.
    """
    payload = {
        key: value
        for key, value in FULL_PAYLOAD.items()
        if key not in {"subjectiveGoals", "objectiveGoals", "patientAdvice"}
    }
    report = score(FirstAssessment.model_validate(payload), [], threshold=THRESHOLD)
    assert report.meetsThreshold


# --------------------------------------------------------------------------
# Rejections
# --------------------------------------------------------------------------
def test_rejections_reduce_confidence():
    assessment = FirstAssessment.model_validate(FULL_PAYLOAD)
    clean = score(assessment, [], threshold=THRESHOLD)
    issue = GroundingIssue(
        path="objectiveGoals[0].targetDate",
        value="2026-09-01",
        reason="date not in transcript",
        overlap=0.0,
    )
    penalised = score(assessment, [issue], threshold=THRESHOLD)

    assert penalised.overall < clean.overall
    assert penalised.rejectedCount == 1


def test_rejected_field_is_flagged_with_the_discarded_value():
    """The audit trail: a clinician must be able to see what was thrown away."""
    issue = GroundingIssue(
        path="clinicalDetails.clinicalHistory",
        value="prior stroke in 2019",
        reason="number(s) not in transcript: 2019",
        overlap=0.1,
    )
    report = score(empty_assessment(), [issue], threshold=THRESHOLD)

    rejected = [f for f in report.flaggedFields if f.reason == "rejected"]
    assert len(rejected) == 1
    assert rejected[0].path == "clinicalDetails.clinicalHistory"
    assert "prior stroke in 2019" in rejected[0].detail
    assert "2019" in rejected[0].detail


def test_a_rejected_field_is_not_also_flagged_as_not_stated():
    """It is blank *because* it was rejected; reporting both would mislead."""
    issue = GroundingIssue(
        path="clinicalDetails.duration", value="six weeks", reason="ungrounded", overlap=0.0
    )
    report = score(empty_assessment(), [issue], threshold=THRESHOLD)

    duration_flags = [f for f in report.flaggedFields if f.path == "clinicalDetails.duration"]
    assert len(duration_flags) == 1
    assert duration_flags[0].reason == "rejected"


def test_rejection_penalty_is_capped():
    """Many rejections must not drive the score negative."""
    issues = [
        GroundingIssue(path=f"f{i}", value="x", reason="ungrounded", overlap=0.0)
        for i in range(20)
    ]
    report = score(FirstAssessment.model_validate(FULL_PAYLOAD), issues, threshold=THRESHOLD)
    assert report.overall >= 0.0


# --------------------------------------------------------------------------
# Flag reporting
# --------------------------------------------------------------------------
def test_blank_clinical_fields_are_flagged_individually():
    report = score(empty_assessment(), [], threshold=THRESHOLD)
    paths = {flag.path for flag in report.flaggedFields}
    assert "clinicalDetails.chiefComplaint" in paths
    assert "clinicalDetails.clinicalHistory" in paths
    assert "clinicalDetails.duration" in paths


def test_absent_sections_are_flagged_once_not_per_field():
    """Per-leaf flags on empty arrays would bury the real signal in noise."""
    report = score(empty_assessment(), [], threshold=THRESHOLD)
    section_flags = [f for f in report.flaggedFields if f.path == "subjectiveGoals"]
    assert len(section_flags) == 1
    assert section_flags[0].detail == "no entries extracted"


def test_missing_target_dates_are_flagged_per_goal():
    """The supplied recording has no dates, so each goal must say so."""
    assessment = FirstAssessment.model_validate(
        {
            "objectiveGoals": [
                {"goalName": "Restore extension", "targetDate": ""},
                {"goalName": "Improve stability", "targetDate": ""},
            ]
        }
    )
    report = score(assessment, [], threshold=THRESHOLD)
    paths = {f.path for f in report.flaggedFields}
    assert "objectiveGoals[0].targetDate" in paths
    assert "objectiveGoals[1].targetDate" in paths


def test_test_without_a_measurement_is_flagged():
    assessment = FirstAssessment.model_validate(
        {"objectiveAssessment": {"tests": [{"testName": "Knee flexion"}]}}
    )
    report = score(assessment, [], threshold=THRESHOLD)
    paths = {f.path for f in report.flaggedFields}
    assert "objectiveAssessment.tests[0]" in paths


def test_report_carries_the_threshold_it_was_judged_against():
    """The API returns this, so a 422 can explain what bar was missed."""
    report = score(empty_assessment(), [], threshold=0.8)
    assert report.threshold == 0.8


def test_an_empty_goal_bucket_is_not_flagged_when_the_other_is_full():
    """Goals are one concern in two buckets, split by whether they measure.

    Reclassifying a narrative goal into subjectiveGoals leaves objectiveGoals
    empty by design; flagging that reports a gap that does not exist.
    """
    assessment = FirstAssessment.model_validate(
        {"subjectiveGoals": [{"goalDetails": "Improve ankle mobility", "targetDate": ""}]}
    )
    report = score(assessment, [], threshold=THRESHOLD)
    paths = {f.path for f in report.flaggedFields}
    assert "objectiveGoals" not in paths
    assert "subjectiveGoals" not in paths


def test_goals_are_still_flagged_when_neither_bucket_has_any():
    report = score(empty_assessment(), [], threshold=THRESHOLD)
    paths = {f.path for f in report.flaggedFields}
    assert "subjectiveGoals" in paths


# --------------------------------------------------------------------------
# False negatives: measurements the recording states but the record missed
# --------------------------------------------------------------------------
from app.extraction.confidence import find_missed_measurements   # noqa: E402

TRANSCRIPT_WITH_FIVE = (
    "left knee flexion of 124° compared with 130° on the right, "
    "left knee extension of 20° compared with 5° on the right, "
    "hip internal rotation of 45° bilaterally, "
    "hip external rotation of 60° bilaterally and "
    "ankle dorsiflexion of 4.5° on the left compared with 12° on the right."
)


def test_a_spoken_measurement_missing_from_the_record_is_detected():
    """Grounding is blind to omissions; this is the other half of the check.

    The reference model emitted eight tests and dropped hip external rotation
    entirely. Nothing flagged it, and the record simply looked complete.
    """
    assessment = FirstAssessment.model_validate(
        {"objectiveAssessment": {"tests": [
            {"testName": "Knee flexion", "left": "124", "right": "130"},
            {"testName": "Knee extension", "left": "20", "right": "5"},
            {"testName": "Hip internal rotation", "left": "45", "right": "45"},
            {"testName": "Ankle dorsiflexion", "left": "4.5", "right": "12"},
        ]}}
    )
    assert find_missed_measurements(TRANSCRIPT_WITH_FIVE, assessment) == ["60"]


def test_a_complete_record_reports_nothing_missed():
    assessment = FirstAssessment.model_validate(
        {"objectiveAssessment": {"tests": [
            {"testName": "Knee flexion", "left": "124", "right": "130"},
            {"testName": "Knee extension", "left": "20", "right": "5"},
            {"testName": "Hip internal rotation", "left": "45", "right": "45"},
            {"testName": "Hip external rotation", "left": "60", "right": "60"},
            {"testName": "Ankle dorsiflexion", "left": "4.5", "right": "12"},
        ]}}
    )
    assert find_missed_measurements(TRANSCRIPT_WITH_FIVE, assessment) == []


def test_missed_measurements_become_flags_not_values():
    """It reports the gap. It never guesses which test the number belonged to."""
    assessment = FirstAssessment.model_validate(
        {"objectiveAssessment": {"tests": [{"testName": "Knee flexion", "left": "124", "right": "130"}]}}
    )
    report = score(assessment, [], threshold=THRESHOLD, transcript=TRANSCRIPT_WITH_FIVE)

    missed = [f for f in report.flaggedFields if f.reason == "possibly_missed"]
    assert {f.detail.split()[0] for f in missed} == {"4.5", "5", "20", "45", "60", "12"} - {"124", "130"}
    assert all(f.path == "objectiveAssessment.tests" for f in missed)


def test_the_check_is_silent_without_a_transcript():
    """Scoring is called in tests and tools with no transcript to hand."""
    report = score(empty_assessment(), [], threshold=THRESHOLD)
    assert not [f for f in report.flaggedFields if f.reason == "possibly_missed"]


def test_a_value_carrying_its_unit_still_counts_as_captured():
    """Regression: this produced a false "everything is missing" report.

    The model stores "124 degrees" rather than "124". A strict float() parsed
    every captured value as nothing, so the completeness check reported all
    eight spoken measurements as missed - including the four sitting in the
    record right next to it.
    """
    assessment = FirstAssessment.model_validate(
        {"objectiveAssessment": {"tests": [
            {"testName": "Knee flexion", "unitName": "degrees", "left": "124\u00b0", "right": "130\u00b0"},
            {"testName": "Knee extension", "unitName": "degrees", "left": "20\u00b0", "right": "5\u00b0"},
            {"testName": "Hip internal rotation", "unitName": "degrees", "left": "45\u00b0", "right": "45\u00b0"},
            {"testName": "Hip external rotation", "unitName": "degrees", "left": "60\u00b0", "right": "60\u00b0"},
            {"testName": "Ankle dorsiflexion", "unitName": "degrees", "left": "4.5\u00b0", "right": "12\u00b0"},
        ]}}
    )
    assert find_missed_measurements(TRANSCRIPT_WITH_FIVE, assessment) == []


def test_the_detector_reads_through_any_unit_spelling():
    from app.extraction.confidence import _as_number

    assert _as_number("124") == "124"
    assert _as_number("124\u00b0") == "124"
    assert _as_number(" 4.5\u00b0 ") == "4.5"
    assert _as_number("20 degrees") == "20"
    assert _as_number("degrees") == ""       # no number at all
