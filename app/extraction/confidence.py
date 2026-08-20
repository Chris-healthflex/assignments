"""Confidence scoring and field flagging - requirement S5.

Two different things get reported, and conflating them would mislead a
clinician:

* **not_stated**  the transcript never covered this field. Expected, benign,
  and the correct outcome for a recording that simply did not discuss goals.
* **rejected**    the model produced a value that failed grounding and was
  cleared. This is a caught hallucination, and it is a much stronger signal
  that the extraction should not be trusted.

Section weights reflect clinical importance rather than field count. A first
assessment without a chief complaint is not usable; one without patient advice
usually is. Weighting by field count instead would let an absent complaint be
offset by a well-populated goals list.
"""

from __future__ import annotations

import re
from typing import Iterable

from pydantic import BaseModel, Field

from app.extraction.grounding import GroundingIssue
from app.schemas.first_assessment import FirstAssessment

#: Contribution of each section to the overall score. Sums to 1.0.
SECTION_WEIGHTS: dict[str, float] = {
    "clinicalDetails": 0.30,
    "objectiveAssessment": 0.25,
    "subjectiveAssessments": 0.15,
    "recommendation": 0.15,
    "subjectiveGoals": 0.05,
    "objectiveGoals": 0.05,
    "patientAdvice": 0.05,
}

#: How much each caught hallucination lowers the overall score. A rejection
#: means the model was willing to invent, so later fields deserve less trust.
REJECTION_PENALTY = 0.10

#: A number carrying a unit is almost always a clinical measurement. Used to
#: check the extraction against the recording for values it simply missed.
_SPOKEN_MEASUREMENT = re.compile(r"(\d+(?:\.\d+)?)\s*(?:°|degrees|deg)", re.IGNORECASE)


_LEADING_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


def _as_number(text: str) -> str:
    """The numeric part of a value, ignoring any unit attached to it.

    A strict float() here caused a false-positive storm: the model stores
    "124 degrees" rather than "124", every captured value parsed as nothing,
    and the completeness check reported every spoken measurement as missing -
    including the ones sitting in the record.
    """
    match = _LEADING_NUMBER.search(str(text or ""))
    if not match:
        return ""
    value = float(match.group(0))
    return str(int(value)) if value.is_integer() else str(value)


def find_missed_measurements(transcript: str, assessment: FirstAssessment) -> list[str]:
    """Measurements spoken in the recording that never reached the record.

    Grounding is one-sided: it catches values the model invented, and is blind
    to values it silently dropped. On the reference recording the model emitted
    eight tests and omitted "hip external rotation of 60 degrees bilaterally"
    entirely - nothing anywhere flagged it, and the record simply looked
    complete.

    Comparing the numbers spoken against the numbers captured closes that gap.
    It reports; it never fills anything in, because which test a loose number
    belongs to is exactly the guess this pipeline refuses to make.
    """
    spoken = {
        _as_number(match.group(1)) for match in _SPOKEN_MEASUREMENT.finditer(transcript or "")
    }
    spoken.discard("")

    captured: set[str] = set()
    for test in assessment.objectiveAssessment.tests:
        for raw in (test.value, test.left, test.right):
            number = _as_number((raw or "").strip())
            if number:
                captured.add(number)

    return sorted(spoken - captured, key=lambda n: float(n))


class FieldFlag(BaseModel):
    """One field a clinician should check before signing."""

    path: str
    reason: str          # "not_stated" | "rejected"
    detail: str = ""


class ConfidenceReport(BaseModel):
    overall: float
    meetsThreshold: bool
    threshold: float
    sectionScores: dict[str, float] = Field(default_factory=dict)
    flaggedFields: list[FieldFlag] = Field(default_factory=list)
    rejectedCount: int = 0

    @property
    def rejected_paths(self) -> list[str]:
        return [f.path for f in self.flaggedFields if f.reason == "rejected"]


def _filled(value: str) -> bool:
    return bool((value or "").strip())


def _score_sections(assessment: FirstAssessment) -> dict[str, float]:
    """Per-section completeness in [0, 1]."""
    details = assessment.clinicalDetails
    clinical = sum(
        _filled(v) for v in (details.clinicalHistory, details.chiefComplaint, details.duration)
    ) / 3

    subjective = (
        1.0
        if any(_filled(a.conclusion) or _filled(a.testName) for a in assessment.subjectiveAssessments)
        else 0.0
    )

    # A test is only worth counting if it carries a measurement.
    tests = assessment.objectiveAssessment.tests
    objective = (
        1.0
        if any(_filled(t.value) or _filled(t.left) or _filled(t.right) for t in tests)
        else 0.0
    )

    recommendation = (
        1.0
        if any(_filled(r.sessionType) or _filled(r.sessionFrequency) for r in assessment.recommendation)
        else 0.0
    )

    return {
        "clinicalDetails": round(clinical, 3),
        "objectiveAssessment": objective,
        "subjectiveAssessments": subjective,
        "recommendation": recommendation,
        "subjectiveGoals": 1.0 if any(_filled(g.goalDetails) for g in assessment.subjectiveGoals) else 0.0,
        "objectiveGoals": 1.0 if any(_filled(g.goalName) for g in assessment.objectiveGoals) else 0.0,
        "patientAdvice": 1.0 if _filled(assessment.patientAdvice.adviceDetails) else 0.0,
    }


def _blank_flags(assessment: FirstAssessment, already_flagged: set[str]) -> list[FieldFlag]:
    """Flag empty fields, without repeating anything grounding already rejected.

    Only fields worth a clinician's attention are listed. Every leaf of an
    empty optional array would be noise, so absent sections are reported once
    at section level rather than per field.
    """
    flags: list[FieldFlag] = []

    for name in ("clinicalHistory", "chiefComplaint", "duration"):
        path = f"clinicalDetails.{name}"
        if path not in already_flagged and not _filled(getattr(assessment.clinicalDetails, name)):
            flags.append(FieldFlag(path=path, reason="not_stated"))

    # Goals are one clinical concern split across two buckets by whether they
    # carry a numeric target. Flagging the empty bucket when the other is full
    # reports a gap that does not exist - the goals were recorded, just as the
    # other kind.
    has_goals = bool(assessment.subjectiveGoals or assessment.objectiveGoals)

    section_checks: Iterable[tuple[str, bool]] = (
        ("subjectiveAssessments", bool(assessment.subjectiveAssessments)),
        ("objectiveAssessment.tests", bool(assessment.objectiveAssessment.tests)),
        ("subjectiveGoals", has_goals),
        ("objectiveGoals", has_goals),
        ("recommendation", bool(assessment.recommendation)),
    )
    for path, present in section_checks:
        if not present and path not in already_flagged:
            flags.append(
                FieldFlag(path=path, reason="not_stated", detail="no entries extracted")
            )

    advice_path = "patientAdvice.adviceDetails"
    if advice_path not in already_flagged and not _filled(assessment.patientAdvice.adviceDetails):
        flags.append(FieldFlag(path=advice_path, reason="not_stated"))

    # Blank leaves inside populated arrays are worth naming individually,
    # since a half-filled measurement is easy to miss on review.
    for index, test in enumerate(assessment.objectiveAssessment.tests):
        if not (_filled(test.value) or _filled(test.left) or _filled(test.right)):
            path = f"objectiveAssessment.tests[{index}]"
            if path not in already_flagged:
                flags.append(
                    FieldFlag(path=path, reason="not_stated", detail="no measurement captured")
                )

    for index, goal in enumerate(assessment.objectiveGoals):
        path = f"objectiveGoals[{index}].targetDate"
        if path not in already_flagged and not _filled(goal.targetDate):
            flags.append(FieldFlag(path=path, reason="not_stated"))

    for index, goal in enumerate(assessment.subjectiveGoals):
        path = f"subjectiveGoals[{index}].targetDate"
        if path not in already_flagged and not _filled(goal.targetDate):
            flags.append(FieldFlag(path=path, reason="not_stated"))

    return flags


def score(
    assessment: FirstAssessment,
    issues: list[GroundingIssue],
    *,
    threshold: float,
    transcript: str = "",
) -> ConfidenceReport:
    """Build the confidence report for a completed extraction."""
    section_scores = _score_sections(assessment)

    weighted = sum(
        section_scores[name] * weight for name, weight in SECTION_WEIGHTS.items()
    )

    rejection_flags = [
        FieldFlag(
            path=issue.path,
            reason="rejected",
            detail=f"value {issue.value!r} discarded: {issue.reason}",
        )
        for issue in issues
    ]
    rejected_paths = {flag.path for flag in rejection_flags}

    penalty = min(len(rejection_flags) * REJECTION_PENALTY, 0.5)
    overall = max(0.0, min(1.0, weighted - penalty))

    flags = rejection_flags + _blank_flags(assessment, rejected_paths)

    # False negatives: spoken measurements that never reached the record.
    for number in find_missed_measurements(transcript, assessment):
        flags.append(
            FieldFlag(
                path="objectiveAssessment.tests",
                reason="possibly_missed",
                detail=f"{number} degrees is stated in the recording but appears in no measurement",
            )
        )

    return ConfidenceReport(
        overall=round(overall, 3),
        meetsThreshold=overall >= threshold,
        threshold=threshold,
        sectionScores=section_scores,
        flaggedFields=flags,
        rejectedCount=len(rejection_flags),
    )
