"""LangGraph-backed, evidence-only clinical extraction."""
from __future__ import annotations

import os
from typing import TypedDict

from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from .schemas import ExtractionEnvelope


class ExtractionError(RuntimeError):
    pass


class ExtractionState(TypedDict):
    transcript: str
    result: ExtractionEnvelope


SYSTEM_PROMPT = """You extract first-assessment data from a clinical transcript.
Return only facts explicitly stated in the transcript. Never infer, calculate,
or invent diagnoses, clinical values, scores, dates, recommendations, or facts.
All schema strings must be strings. Use an empty string when a string is not
supported. Use empty arrays when no array item is supported. For every omitted
or ambiguous clinically meaningful field, add its dotted schema path and a
short reason to uncertain_fields. Do not flag intentionally empty optional
arrays. Keep evidence wording concise and faithful to the transcript."""


def _extract(state: ExtractionState) -> dict:
    if not os.getenv("OPENAI_API_KEY"):
        raise ExtractionError("OPENAI_API_KEY is required for structured extraction.")
    try:
        model = ChatOpenAI(model=os.getenv("EXTRACTION_MODEL", "gpt-4o-mini"), temperature=0)
        extractor = model.with_structured_output(ExtractionEnvelope)
        result = extractor.invoke(
            [("system", SYSTEM_PROMPT), ("human", state["transcript"])]
        )
        return {"result": result}
    except Exception as exc:
        if isinstance(exc, ExtractionError):
            raise
        raise ExtractionError(f"Clinical extraction failed: {exc}") from exc


def build_extraction_graph():
    graph = StateGraph(ExtractionState)
    graph.add_node("extract_evidence_only", _extract)
    graph.add_edge(START, "extract_evidence_only")
    graph.add_edge("extract_evidence_only", END)
    return graph.compile()


def extract_assessment(transcript: str) -> ExtractionEnvelope:
    if not transcript.strip():
        raise ExtractionError("Transcript is empty.")
    result = build_extraction_graph().invoke({"transcript": transcript})
    return result["result"]
