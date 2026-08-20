"""LangGraph extraction agent: transcript -> validated ExtractionPayload.

Graph shape:

    extract --> validate --+--> (ok / out of retries) --> END
       ^                   |
       +---- repair <------+   (schema violation, bounded retries)

The repair edge is the point of the whole agent: a single-shot prompt that
mostly works is not good enough for clinical data, so a failed validation is
fed back to the model as an explicit error list rather than silently dropped.
"""

from __future__ import annotations

import logging
from typing import Any, TypedDict

from pydantic import ValidationError

from app.config import get_settings
from app.schemas import ExtractionPayload

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a clinical scribe assistant. You convert a \
transcript of a physiotherapy first-assessment consultation into structured \
data.

Rules:
- Extract only what the transcript states or clearly implies. Never infer a \
diagnosis, a pain score, or a history item that was not discussed.
- If a field is not supported by the transcript, leave it unset and add its \
dotted path to `unresolved_fields`.
- Preserve the patient's own wording for durations and free-text fields.
- Numeric pain scores are 0-10 and must come from the patient, not your \
judgement."""


class ExtractionState(TypedDict, total=False):
    """State threaded through the graph."""

    transcript: str
    payload: ExtractionPayload | None
    raw: dict[str, Any] | None
    errors: list[str]
    attempts: int


class ExtractionFailed(RuntimeError):
    """Raised when the agent cannot produce a schema-valid payload."""


# --------------------------------------------------------------------------- #
# Nodes
# --------------------------------------------------------------------------- #
def _client():
    import anthropic  # imported lazily so tests can run without the SDK

    settings = get_settings()
    return anthropic.Anthropic(api_key=settings.anthropic_api_key)


def _call_model(messages: list[dict[str, Any]]) -> ExtractionPayload:
    settings = get_settings()
    response = _client().messages.parse(
        model=settings.extraction_model,
        max_tokens=16000,
        system=SYSTEM_PROMPT,
        thinking={"type": "adaptive"},
        messages=messages,
        output_format=ExtractionPayload,
    )
    return response.parsed_output


def extract_node(state: ExtractionState) -> ExtractionState:
    """First pass: transcript in, structured payload out."""
    transcript = state["transcript"]
    messages = [
        {
            "role": "user",
            "content": f"<transcript>\n{transcript}\n</transcript>\n\n"
            "Extract the first assessment.",
        }
    ]
    try:
        payload = _call_model(messages)
        return {"payload": payload, "errors": [], "attempts": state.get("attempts", 0) + 1}
    except ValidationError as exc:
        return {
            "payload": None,
            "errors": [str(exc)],
            "attempts": state.get("attempts", 0) + 1,
        }


def validate_node(state: ExtractionState) -> ExtractionState:
    """Re-validate defensively and collect business-rule violations.

    Structured outputs already enforce the JSON schema; this node exists for
    the rules the schema cannot express (e.g. exactly one primary complaint).
    """
    payload = state.get("payload")
    if payload is None:
        return {"errors": state.get("errors") or ["No payload produced."]}

    errors: list[str] = []
    primary = [c for c in payload.complaints if c.is_primary]
    if payload.complaints and len(primary) != 1:
        errors.append(
            f"Expected exactly one primary complaint, found {len(primary)}."
        )
    return {"errors": errors}


def repair_node(state: ExtractionState) -> ExtractionState:
    """Feed the validation errors back to the model for a corrected pass."""
    errors = "\n".join(f"- {e}" for e in state.get("errors", []))
    messages = [
        {
            "role": "user",
            "content": f"<transcript>\n{state['transcript']}\n</transcript>\n\n"
            "Your previous extraction was rejected:\n"
            f"{errors}\n\n"
            "Produce a corrected extraction. Do not invent data to satisfy a "
            "rule -- if the transcript does not support a field, leave it unset.",
        }
    ]
    try:
        payload = _call_model(messages)
        return {"payload": payload, "errors": [], "attempts": state["attempts"] + 1}
    except ValidationError as exc:
        return {"errors": [str(exc)], "attempts": state["attempts"] + 1}


def _route(state: ExtractionState) -> str:
    """Continue to repair only while there are errors and retries left."""
    settings = get_settings()
    if not state.get("errors"):
        return "done"
    if state.get("attempts", 0) > settings.extraction_max_retries:
        return "done"
    return "repair"


# --------------------------------------------------------------------------- #
# Graph
# --------------------------------------------------------------------------- #
def build_graph():
    from langgraph.graph import END, StateGraph

    graph = StateGraph(ExtractionState)
    graph.add_node("extract", extract_node)
    graph.add_node("validate", validate_node)
    graph.add_node("repair", repair_node)

    graph.set_entry_point("extract")
    graph.add_edge("extract", "validate")
    graph.add_conditional_edges("validate", _route, {"repair": "repair", "done": END})
    graph.add_edge("repair", "validate")
    return graph.compile()


def extract(transcript: str) -> tuple[ExtractionPayload, list[str]]:
    """Run the agent. Returns (payload, remaining_errors)."""
    if not transcript.strip():
        raise ExtractionFailed("Empty transcript.")

    result = build_graph().invoke({"transcript": transcript, "attempts": 0})
    payload = result.get("payload")
    if payload is None:
        raise ExtractionFailed(
            "Extraction produced no valid payload: " + "; ".join(result.get("errors", []))
        )
    return payload, result.get("errors", [])
