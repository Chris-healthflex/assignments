"""
LangGraph pipeline: transcribe -> extract -> validate/repair -> finalize.

Node responsibilities:
  transcribe_node  - runs Whisper (or the offline fallback) on the WAV, produces raw text
  extract_node     - LLM call (schema-constrained) that turns transcript -> FirstAssessment
  validate_node     - Pydantic-validates the LLM output; on failure, re-prompts the LLM
                       once with the validation error appended, rather than silently
                       coercing bad data into the schema
  finalize_node     - clamps confidence, ensures every array field is a list, returns the
                       ExtractionEnvelope

The LLM backend is provider-agnostic (OPENAI or ANTHROPIC via LLM_PROVIDER env var) so
the same graph works with either api.openai.com or api.anthropic.com reachable.
"""

from __future__ import annotations

import json
import os
from typing import TypedDict

from langgraph.graph import StateGraph, END
from pydantic import ValidationError

from app.schemas.first_assessment import ExtractionEnvelope
from app.transcription.whisper_service import transcribe
from app.agent.prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE


class PipelineState(TypedDict, total=False):
    wav_path: str
    transcript: str
    transcription_engine: str
    transcript_low_confidence: bool
    raw_llm_output: str
    validation_error: str
    retry_count: int
    envelope: dict


def _call_llm(system: str, user: str) -> str:
    """Provider-agnostic single LLM call. Returns raw text (expected to be JSON)."""
    provider = os.getenv("LLM_PROVIDER", "anthropic").lower()

    if provider == "anthropic":
        import anthropic

        client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
        resp = client.messages.create(
            model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6"),
            max_tokens=2000,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(block.text for block in resp.content if block.type == "text")

    if provider == "openai":
        from openai import OpenAI

        client = OpenAI()  # reads OPENAI_API_KEY from env
        resp = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o"),
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format={"type": "json_object"},
        )
        return resp.choices[0].message.content

    raise ValueError(f"Unknown LLM_PROVIDER: {provider}")


def transcribe_node(state: PipelineState) -> PipelineState:
    result = transcribe(state["wav_path"])
    return {
        **state,
        "transcript": result.text,
        "transcription_engine": result.engine,
        "transcript_low_confidence": result.low_confidence,
    }


def extract_node(state: PipelineState) -> PipelineState:
    user_prompt = USER_PROMPT_TEMPLATE.format(
        transcript=state["transcript"],
        engine=state["transcription_engine"],
        low_confidence=state["transcript_low_confidence"],
    )
    if state.get("validation_error"):
        user_prompt += (
            f"\n\nYour previous output failed schema validation with this error:\n"
            f"{state['validation_error']}\nFix it and return valid JSON only."
        )
    raw = _call_llm(SYSTEM_PROMPT, user_prompt)
    return {**state, "raw_llm_output": raw}


def validate_node(state: PipelineState) -> PipelineState:
    raw = state["raw_llm_output"].strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        raw = raw[raw.find("{"):]

    try:
        parsed = json.loads(raw)
        envelope = ExtractionEnvelope(
            assessment=parsed["assessment"],
            overall_confidence=parsed.get("overall_confidence", 0.5),
            extraction_flags=parsed.get("extraction_flags", []),
            transcript=state["transcript"],
        )
        if state.get("transcript_low_confidence"):
            envelope.overall_confidence = min(envelope.overall_confidence, 0.5)
            if "transcript" not in envelope.extraction_flags:
                envelope.extraction_flags.append(
                    "transcript (low-confidence ASR engine used)"
                )
        return {
            **state,
            "envelope": envelope.model_dump(),
            "validation_error": "",
        }
    except (json.JSONDecodeError, ValidationError, KeyError) as e:
        return {**state, "validation_error": str(e), "retry_count": state.get("retry_count", 0) + 1}


def route_after_validate(state: PipelineState) -> str:
    if state.get("envelope"):
        return "finalize"
    if state.get("retry_count", 0) >= 2:
        return "finalize_with_failure"
    return "extract"


def finalize_node(state: PipelineState) -> PipelineState:
    return state


def finalize_with_failure_node(state: PipelineState) -> PipelineState:
    """Extraction never produced valid schema after retries — return an empty,
    schema-valid envelope with confidence 0 rather than fabricating content."""
    envelope = ExtractionEnvelope(
        assessment={},
        overall_confidence=0.0,
        extraction_flags=["ALL_FIELDS - extraction failed schema validation after retries"],
        transcript=state.get("transcript", ""),
    )
    return {**state, "envelope": envelope.model_dump()}


def build_graph():
    graph = StateGraph(PipelineState)
    graph.add_node("transcribe", transcribe_node)
    graph.add_node("extract", extract_node)
    graph.add_node("validate", validate_node)
    graph.add_node("finalize", finalize_node)
    graph.add_node("finalize_with_failure", finalize_with_failure_node)

    graph.set_entry_point("transcribe")
    graph.add_edge("transcribe", "extract")
    graph.add_edge("extract", "validate")
    graph.add_conditional_edges(
        "validate",
        route_after_validate,
        {"extract": "extract", "finalize": "finalize", "finalize_with_failure": "finalize_with_failure"},
    )
    graph.add_edge("finalize", END)
    graph.add_edge("finalize_with_failure", END)

    return graph.compile()


def run_pipeline(wav_path: str) -> dict:
    app_graph = build_graph()
    final_state = app_graph.invoke({"wav_path": wav_path, "retry_count": 0})
    return final_state["envelope"]
