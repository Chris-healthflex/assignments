"""Shared state passed between LangGraph nodes."""
from __future__ import annotations

from typing import Any, Dict, List, TypedDict


class ExtractionState(TypedDict, total=False):
    transcript: str

    # raw per-section extractions (dicts / lists straight from the LLM or stub)
    clinicalDetails: Dict[str, Any]
    subjectiveAssessments: List[Dict[str, Any]]
    objectiveTests: List[Dict[str, Any]]
    subjectiveGoals: List[Dict[str, Any]]
    objectiveGoals: List[Dict[str, Any]]
    recommendation: List[Dict[str, Any]]
    patientAdvice: Dict[str, Any]

    # per-node timings (seconds)
    timings: Dict[str, float]
