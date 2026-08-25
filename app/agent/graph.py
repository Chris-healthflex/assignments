"""LangGraph workflow: extract -> validate -> confidence gate."""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import ValidationError

from app.config import get_settings
from app.models.assessment import SECTION_ALIASES, FirstAssessment
from app.models.internal import ExtractionEnvelope, LowConfidenceField, PipelineResult
from app.agent.prompts import SYSTEM_PROMPT, USER_PROMPT

logger = logging.getLogger(__name__)


class GraphState(TypedDict, total=False):
    transcript: str
    envelope: Optional[ExtractionEnvelope]
    assessment: Optional[FirstAssessment]
    field_confidence: Dict[str, float]
    low_confidence_fields: List[LowConfidenceField]
    error: Optional[str]


#: Sensible default model per provider, used when LLM_MODEL is blank.
_PROVIDER_DEFAULTS = {
    "openai": "gpt-4o-mini",
    "google": "gemini-3.6-flash",
    "groq": "llama-3.3-70b-versatile",
}


def _default_llm():
    """Build the structured-output LLM for the configured provider.

    Provider packages are imported lazily so that (a) the test suite never
    needs a key or a network call, and (b) you only need the SDK for the
    provider you actually use.
    """
    settings = get_settings()
    provider = (settings.llm_provider or "openai").strip().lower()
    model = settings.llm_model or _PROVIDER_DEFAULTS.get(provider, "")

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        if not settings.openai_api_key:
            raise RuntimeError("LLM_PROVIDER=openai but OPENAI_API_KEY is not set")
        llm = ChatOpenAI(
            model=model,
            temperature=settings.llm_temperature,
            api_key=settings.openai_api_key,
        )
    elif provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        if not settings.google_api_key:
            raise RuntimeError("LLM_PROVIDER=google but GOOGLE_API_KEY is not set")
        llm = ChatGoogleGenerativeAI(
            model=model,
            temperature=settings.llm_temperature,
            google_api_key=settings.google_api_key,
        )
    elif provider == "groq":
        from langchain_groq import ChatGroq

        if not settings.groq_api_key:
            raise RuntimeError("LLM_PROVIDER=groq but GROQ_API_KEY is not set")
        llm = ChatGroq(
            model=model,
            temperature=settings.llm_temperature,
            api_key=settings.groq_api_key,
        )
    else:
        raise RuntimeError(
            f"Unknown LLM_PROVIDER '{provider}'. Use one of: "
            + ", ".join(_PROVIDER_DEFAULTS)
        )

    logger.info("Using LLM provider '%s' model '%s'", provider, model)
    return llm.with_structured_output(ExtractionEnvelope)


def build_graph(llm: Any = None, threshold: Optional[float] = None):
    """Compile the extraction graph.

    `llm` must be a structured-output runnable returning an ExtractionEnvelope.
    Injecting it keeps the graph testable without network access.
    """
    settings = get_settings()
    effective_threshold = settings.confidence_threshold if threshold is None else threshold

    def extract_clinical_data(state: GraphState) -> GraphState:
        transcript = (state.get("transcript") or "").strip()
        if not transcript:
            return {"error": "Transcript is empty; nothing to extract."}

        runnable = llm if llm is not None else _default_llm()
        messages = [
            ("system", SYSTEM_PROMPT),
            ("human", USER_PROMPT.format(transcript=transcript)),
        ]
        try:
            envelope = runnable.invoke(messages)
        except Exception as exc:
            logger.exception("LLM extraction failed")
            return {"error": f"Clinical extraction failed: {exc}"}

        if isinstance(envelope, dict):
            try:
                envelope = ExtractionEnvelope.model_validate(envelope)
            except ValidationError as exc:
                return {"error": f"Model returned malformed output: {exc}"}

        return {"envelope": envelope}

    def validate_assessment(state: GraphState) -> GraphState:
        envelope = state.get("envelope")
        if envelope is None:
            return {"error": state.get("error") or "No extraction result produced."}
        try:
            assessment = FirstAssessment.model_validate(
                envelope.assessment.model_dump()
            )
        except ValidationError as exc:
            return {"error": f"Assessment failed schema validation: {exc}"}
        return {
            "assessment": assessment,
            "field_confidence": dict(envelope.field_confidence),
        }

    def confidence_check(state: GraphState) -> GraphState:
        if state.get("error"):
            return {}
        scores = state.get("field_confidence") or {}
        failures: List[LowConfidenceField] = []
        for field_name, alias in SECTION_ALIASES.items():
            # Accept either the camelCase alias or the snake_case attribute name.
            score = scores.get(alias, scores.get(field_name))
            if score is None:
                # Absent score means the model never rated it - treat as unknown,
                # not as a pass.
                score = 0.0
            if score < effective_threshold:
                failures.append(
                    LowConfidenceField(
                        field=alias,
                        confidence=round(float(score), 3),
                        threshold=effective_threshold,
                    )
                )
        return {"low_confidence_fields": failures}

    def _after_extract(state: GraphState) -> str:
        return "error" if state.get("error") else "validate_assessment"

    def _after_validate(state: GraphState) -> str:
        return "error" if state.get("error") else "confidence_check"

    graph = StateGraph(GraphState)
    graph.add_node("extract_clinical_data", extract_clinical_data)
    graph.add_node("validate_assessment", validate_assessment)
    graph.add_node("confidence_check", confidence_check)

    graph.add_edge(START, "extract_clinical_data")
    graph.add_conditional_edges(
        "extract_clinical_data",
        _after_extract,
        {"validate_assessment": "validate_assessment", "error": END},
    )
    graph.add_conditional_edges(
        "validate_assessment",
        _after_validate,
        {"confidence_check": "confidence_check", "error": END},
    )
    graph.add_edge("confidence_check", END)
    return graph.compile()


def run_extraction(
    transcript: str, llm: Any = None, threshold: Optional[float] = None
) -> PipelineResult:
    """Run the graph over a transcript and normalise the output."""
    compiled = build_graph(llm=llm, threshold=threshold)
    final: Dict[str, Any] = compiled.invoke({"transcript": transcript})
    return PipelineResult(
        assessment=final.get("assessment"),
        field_confidence=final.get("field_confidence") or {},
        low_confidence_fields=final.get("low_confidence_fields") or [],
        transcript=transcript,
        error=final.get("error"),
    )
