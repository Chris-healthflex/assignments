"""Confidence scoring + flagging.

Per-section score reflects how completely the section was filled from stated content.
`flaggedFields` records every empty scalar / empty list as `not_stated`, and carries
forward the `ungrounded` flags from the grounding stage. The overall score is the
mean of section scores; below `threshold` the whole result is marked as not meeting
the bar (the caller can then reject or route for human review).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from app.extraction.grounding import GroundingReport
from app.schemas.assessment import FirstAssessment


@dataclass
class ConfidenceResult:
    overall: float
    threshold: float
    meets_threshold: bool
    section_scores: Dict[str, float] = field(default_factory=dict)
    rejected_count: int = 0
    flagged: List[dict] = field(default_factory=list)


def _scalar_score(*values: str) -> float:
    filled = sum(1 for v in values if v and v.strip())
    return filled / len(values) if values else 0.0


def _list_score(items: list) -> float:
    return 1.0 if items else 0.0


def score(
    assessment: FirstAssessment,
    grounding: GroundingReport,
    threshold: float,
) -> ConfidenceResult:
    cd = assessment.clinicalDetails
    # Objective section is scored on measurement COVERAGE (from grounding), not on
    # "is the list non-empty" — so a table that drops or duplicates a measurement
    # can never report 1.0. Falls back to list presence only if coverage wasn't run.
    objective_score = grounding.coverage.get(
        "objectiveAssessment", _list_score(assessment.objectiveAssessment.tests)
    )
    section_scores: Dict[str, float] = {
        "clinicalDetails": _scalar_score(cd.clinicalHistory, cd.chiefComplaint, cd.duration),
        "subjectiveAssessments": _list_score(assessment.subjectiveAssessments),
        "objectiveAssessment": objective_score,
        "subjectiveGoals": _list_score(assessment.subjectiveGoals),
        "objectiveGoals": _list_score(assessment.objectiveGoals),
        "recommendation": _list_score(assessment.recommendation),
        "patientAdvice": _scalar_score(assessment.patientAdvice.adviceDetails),
    }
    overall = round(sum(section_scores.values()) / len(section_scores), 2)

    flagged: List[dict] = []

    # not_stated: empty scalar leaves
    if not cd.clinicalHistory:
        flagged.append({"path": "clinicalDetails.clinicalHistory", "reason": "not_stated", "detail": ""})
    if not cd.chiefComplaint:
        flagged.append({"path": "clinicalDetails.chiefComplaint", "reason": "not_stated", "detail": ""})
    if not cd.duration:
        flagged.append({"path": "clinicalDetails.duration", "reason": "not_stated", "detail": ""})
    if not assessment.patientAdvice.adviceDetails:
        flagged.append({"path": "patientAdvice.adviceDetails", "reason": "not_stated", "detail": ""})

    # not_stated: empty lists
    if not assessment.subjectiveGoals:
        flagged.append({"path": "subjectiveGoals", "reason": "not_stated", "detail": "no entries extracted"})
    if not assessment.subjectiveAssessments:
        flagged.append({"path": "subjectiveAssessments", "reason": "not_stated", "detail": "no entries extracted"})
    if not assessment.objectiveAssessment.tests:
        flagged.append({"path": "objectiveAssessment.tests", "reason": "not_stated", "detail": "no entries extracted"})
    if not assessment.objectiveGoals:
        flagged.append({"path": "objectiveGoals", "reason": "not_stated", "detail": "no entries extracted"})
    if not assessment.recommendation:
        flagged.append({"path": "recommendation", "reason": "not_stated", "detail": "no entries extracted"})

    # not_stated: empty targetDate on each objective goal (common, worth surfacing)
    for i, g in enumerate(assessment.objectiveGoals):
        if not g.targetDate:
            flagged.append({"path": f"objectiveGoals[{i}].targetDate", "reason": "not_stated", "detail": ""})

    # carry forward grounding flags (ungrounded values + coverage problems)
    flagged.extend(grounding.ungrounded)

    # rejectedCount = values actually blanked as ungrounded (not coverage notices)
    rejected = sum(1 for f in grounding.ungrounded if f["reason"] == "ungrounded")

    return ConfidenceResult(
        overall=overall,
        threshold=threshold,
        meets_threshold=overall >= threshold,
        section_scores=section_scores,
        rejected_count=rejected,
        flagged=flagged,
    )
