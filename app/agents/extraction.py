from __future__ import annotations

import logging
import re
from typing import Annotated, Any, List, Optional, TypedDict

from pydantic import BaseModel, BeforeValidator, Field
from typing_extensions import NotRequired

from app.agents.llm import build_llm
from app.agents.prompts import RETRY_SUFFIX, SYSTEM_PROMPT, USER_PROMPT
from app.config import Settings, get_settings
from app.schemas.api import ExtractionMeta, FieldConfidence
from app.schemas.assessment import (
    LIST_FIELD_PATHS,
    SCALAR_FIELD_PATHS,
    FirstAssessment,
)

logger = logging.getLogger(__name__)

_TRIVIAL_NUMBERS = {"0", "1", "2", "5", "10", "100"}

_NON_RETRYABLE_MARKERS = (
    "rate_limit", "rate limit", "request too large", "413", "429",
    "authentication", "api key", "invalid_api_key", "permission",
    "insufficient_quota", "model_not_found", "does not exist",
)


def _is_retryable(message: str) -> bool:
    """True for schema violations, which feeding the error back can actually fix."""
    low = message.lower()
    return not any(marker in low for marker in _NON_RETRYABLE_MARKERS)


def _clamp01(v: Any) -> Any:
    """Metadata is advisory — clamp rather than reject a stray 1.2."""
    try:
        return min(1.0, max(0.0, float(v)))
    except (TypeError, ValueError):
        return 0.0


class FieldConfidenceOut(BaseModel):
    """Per-field confidence as reported by the model."""

    field: str = Field(description="Dotted schema path, e.g. clinicalDetails.duration")
    confidence: Annotated[float, BeforeValidator(_clamp01)] = 0.0
    evidence: str = Field(default="", description="Short verbatim quote from the transcript")


class ExtractionEnvelope(BaseModel):
    """What the LLM is asked to return: the record plus its own uncertainty."""

    assessment: FirstAssessment = Field(default_factory=FirstAssessment)
    field_confidence: List[FieldConfidenceOut] = Field(default_factory=list)
    unextracted_fields: List[str] = Field(
        default_factory=list,
        description="Dotted paths of fields the transcript does not support",
    )


class ExtractionState(TypedDict):
    """Graph state. Only ``transcript`` is required on entry."""

    transcript: str
    attempts: NotRequired[int]
    error: NotRequired[Optional[str]]
    fatal: NotRequired[bool]
    errors: NotRequired[List[str]]
    draft: NotRequired[Optional[ExtractionEnvelope]]
    assessment: NotRequired[FirstAssessment]
    field_confidence: NotRequired[List[FieldConfidence]]
    unextracted: NotRequired[List[str]]
    warnings: NotRequired[List[str]]
    overall_confidence: NotRequired[float]


def _normalise(text: str) -> str:
    """Lowercase and strip punctuation so digit lookups aren't foiled by commas."""
    return re.sub(r"[^a-z0-9.\s]", " ", text.lower())


def _numbers_in(text: str) -> set[str]:
    return set(re.findall(r"\d+(?:\.\d+)?", text or ""))


def _empty_paths(assessment: FirstAssessment) -> list[str]:
    """Dotted paths that came back empty — computed, not taken on trust."""
    missing: list[str] = []
    dumped = assessment.model_dump()

    for path in SCALAR_FIELD_PATHS:
        section, key = path.split(".")
        if not str(dumped.get(section, {}).get(key, "")).strip():
            missing.append(path)

    for path in LIST_FIELD_PATHS:
        if "." in path:
            section, key = path.split(".")
            items = dumped.get(section, {}).get(key, [])
        else:
            items = dumped.get(path, [])
        if not items:
            missing.append(path)

    return missing


def _make_extract_node(settings: Settings):
    llm = build_llm(settings, ExtractionEnvelope)

    def extract(state: ExtractionState) -> dict:
        attempts = state.get("attempts", 0) + 1
        prompt = USER_PROMPT.format(transcript=state["transcript"])
        if state.get("error"):
            prompt += RETRY_SUFFIX.format(error=state["error"])

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        try:
            draft = llm.invoke(messages)
        except Exception as exc: 
            message = str(exc)[:1500]
            retryable = _is_retryable(message)
            logger.warning(
                "Extraction attempt %s failed (%s): %s",
                attempts, "retryable" if retryable else "fatal", message,
            )
            return {
                "attempts": attempts,
                "error": message,
                "draft": None,
                "fatal": not retryable,
                "errors": state.get("errors", []) + [message],
            }

        return {"attempts": attempts, "error": None, "draft": draft}

    return extract


def ground_check(state: ExtractionState) -> dict:
    draft = state.get("draft")
    if draft is None:
        return {"warnings": []}

    spoken = _numbers_in(_normalise(state["transcript"]))
    warnings: list[str] = []

    def check(label: str, value: str) -> None:
        for num in _numbers_in(value) - _TRIVIAL_NUMBERS:
            if num not in spoken:
                warnings.append(
                    f"{label}: value '{value}' contains '{num}', which does not "
                    f"appear in the transcript — verify before sign-off."
                )

    for i, test in enumerate(draft.assessment.objectiveAssessment.tests):
        for fname in ("value", "left", "right"):
            check(f"objectiveAssessment.tests[{i}].{fname}", getattr(test, fname))

    for i, goal in enumerate(draft.assessment.objectiveGoals):
        check(f"objectiveGoals[{i}].value", goal.value)
        check(f"objectiveGoals[{i}].targetDate", goal.targetDate)

    for i, goal in enumerate(draft.assessment.subjectiveGoals):
        check(f"subjectiveGoals[{i}].targetDate", goal.targetDate)

    return {"warnings": warnings}


def score(state: ExtractionState) -> dict:
    draft = state.get("draft")
    warnings = state.get("warnings", [])

    if draft is None:
        empty = FirstAssessment()
        return {
            "assessment": empty,
            "field_confidence": [],
            "unextracted": _empty_paths(empty),
            "overall_confidence": 0.0,
        }

    assessment = draft.assessment
    empty = _empty_paths(assessment)

    reported = [
        FieldConfidence(field=fc.field, confidence=fc.confidence, evidence=fc.evidence)
        for fc in draft.field_confidence
        if fc.field not in empty
    ]
    mean_conf = sum(fc.confidence for fc in reported) / len(reported) if reported else 0.0

    core = ("clinicalDetails.chiefComplaint", "clinicalDetails.clinicalHistory",
            "clinicalDetails.duration")
    coverage = sum(1 for p in core if p not in empty) / len(core)

    overall = 0.7 * mean_conf + 0.3 * coverage - 0.05 * len(warnings)
    overall = round(min(1.0, max(0.0, overall)), 4)

    unextracted = sorted(set(empty) | {f for f in draft.unextracted_fields if f})

    return {
        "assessment": assessment,
        "field_confidence": reported,
        "unextracted": unextracted,
        "overall_confidence": overall,
    }


def build_graph(settings: Settings | None = None):
    """Compile the extraction graph."""
    from langgraph.graph import END, StateGraph

    settings = settings or get_settings()
    max_attempts = max(1, settings.max_extraction_attempts)

    def route(state: ExtractionState) -> str:
        if state.get("error") is None:
            return "ground_check"
        if state.get("fatal"):
            return "score"
        return "extract" if state.get("attempts", 0) < max_attempts else "score"

    graph = StateGraph(ExtractionState)
    graph.add_node("extract", _make_extract_node(settings))
    graph.add_node("ground_check", ground_check)
    graph.add_node("score", score)

    graph.set_entry_point("extract")
    graph.add_conditional_edges(
        "extract", route, {"extract": "extract", "ground_check": "ground_check", "score": "score"}
    )
    graph.add_edge("ground_check", "score")
    graph.add_edge("score", END)
    return graph.compile()


class ExtractionOutcome(BaseModel):
    """What the API layer receives from the agent."""

    assessment: FirstAssessment
    meta: ExtractionMeta


def run_extraction(
    transcript: str,
    settings: Settings | None = None,
    graph: Any | None = None,
) -> ExtractionOutcome:
    """Run the graph over a transcript and package the result."""
    settings = settings or get_settings()
    graph = graph or build_graph(settings)

    final = graph.invoke({"transcript": transcript})

    return ExtractionOutcome(
        assessment=final.get("assessment") or FirstAssessment(),
        meta=ExtractionMeta(
            transcript=transcript,
            overallConfidence=final.get("overall_confidence", 0.0),
            fieldConfidence=final.get("field_confidence", []),
            unextractedFields=final.get("unextracted", []),
            groundingWarnings=final.get("warnings", []),
            extractionErrors=final.get("errors", []),
            llmProvider=settings.llm_provider,
            llmModel="" if settings.llm_provider == "stub" else settings.llm_model,
            attempts=final.get("attempts", 0),
        ),
    )
