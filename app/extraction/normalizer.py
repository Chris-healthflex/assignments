"""Map raw per-section extractions into a validated FirstAssessment.

Anything the LLM omits becomes the schema default ("" / []). Pydantic then enforces
the contract (`extra="forbid"`), so a stray key from the model raises instead of
silently polluting the frontend payload.
"""
from __future__ import annotations

from typing import Any, Dict

from app.extraction.state import ExtractionState
from app.schemas.assessment import (
    ClinicalDetails,
    FirstAssessment,
    ObjectiveAssessment,
    ObjectiveGoal,
    ObjectiveTest,
    PatientAdvice,
    Recommendation,
    SubjectiveAssessment,
    SubjectiveGoal,
)


def _clean(d: Dict[str, Any], keys: tuple[str, ...]) -> Dict[str, Any]:
    """Keep only known keys so unexpected LLM fields don't trip extra=forbid."""
    return {k: d.get(k, "") for k in keys}


def build_assessment(state: ExtractionState) -> FirstAssessment:
    cd = state.get("clinicalDetails") or {}
    clinical = ClinicalDetails(**_clean(cd, ("clinicalHistory", "chiefComplaint", "duration")))

    subj = [
        SubjectiveAssessment(**_clean(x, ("testName", "conclusion")))
        for x in (state.get("subjectiveAssessments") or [])
    ]

    tests = [
        ObjectiveTest(**_clean(x, ("testName", "unitName", "value", "left", "right", "comments")))
        for x in (state.get("objectiveTests") or [])
    ]
    objective = ObjectiveAssessment(tests=tests)

    sgoals = [
        SubjectiveGoal(**_clean(x, ("goalDetails", "targetDate")))
        for x in (state.get("subjectiveGoals") or [])
    ]
    ogoals = [
        ObjectiveGoal(**_clean(x, ("goalName", "goalCategory", "unitName", "value", "targetDate")))
        for x in (state.get("objectiveGoals") or [])
    ]

    rec = [
        Recommendation(**_clean(x, ("sessionType", "sessionFrequency")))
        for x in (state.get("recommendation") or [])
    ]

    pa = state.get("patientAdvice") or {}
    advice = PatientAdvice(**_clean(pa, ("adviceDetails",)))

    return FirstAssessment(
        clinicalDetails=clinical,
        subjectiveAssessments=subj,
        objectiveAssessment=objective,
        subjectiveGoals=sgoals,
        objectiveGoals=ogoals,
        recommendation=rec,
        patientAdvice=advice,
    )
