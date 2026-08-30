from __future__ import annotations

import inspect
import logging
import re
from collections.abc import Awaitable, Callable
from typing import Any, TypedDict

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.config import Settings
from app.errors import ExtractionConfidenceError, TranscriptionError
from app.pipeline.prompts import EXTRACTION_SYSTEM_PROMPT
from app.schemas.first_assessment import FirstAssessment, normalize_assessment

logger = logging.getLogger(__name__)


class ExtractionResult(BaseModel):
    assessment: FirstAssessment
    confidence: dict[str, float] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")


class ExtractionState(TypedDict, total=False):
    transcript: str
    raw_result: ExtractionResult | dict[str, Any]
    assessment: FirstAssessment
    confidence: dict[str, float]
    low_confidence_fields: list[str]


Extractor = Callable[[str], ExtractionResult | dict[str, Any] | Awaitable[ExtractionResult | dict[str, Any]]]


def has_useful_transcript(text: str) -> bool:
    words = re.findall(r"[A-Za-z0-9]+", text)
    return len(words) >= 3


def build_extraction_graph(settings: Settings, extractor: Extractor | None = None):
    try:
        from langgraph.graph import END, StateGraph
    except ImportError as exc:
        raise TranscriptionError("LangGraph is not installed") from exc

    async def extract_node(state: ExtractionState) -> ExtractionState:
        transcript = state["transcript"].strip()

        if extractor is not None:
            result = extractor(transcript)
            if inspect.isawaitable(result):
                result = await result
            return {"raw_result": result}

        try:
            from langchain_core.messages import HumanMessage, SystemMessage
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise TranscriptionError("LangChain OpenAI dependencies are not installed") from exc

        if not settings.openai_api_key:
            raise TranscriptionError("OPENAI_API_KEY is not configured")

        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=settings.openai_api_key)
        structured_llm = llm.with_structured_output(ExtractionResult)
        result = await structured_llm.ainvoke(
            [
                SystemMessage(content=EXTRACTION_SYSTEM_PROMPT),
                HumanMessage(content=f"Transcript:\n{transcript}"),
            ]
        )
        return {"raw_result": result}

    async def validation_node(state: ExtractionState) -> ExtractionState:
        raw_result = state["raw_result"]

        try:
            if isinstance(raw_result, ExtractionResult):
                result = raw_result
            else:
                result = ExtractionResult.model_validate(raw_result)

            assessment = normalize_assessment(result.assessment.model_dump())
        except ValidationError:
            logger.info("LLM returned assessment data that failed validation")
            raise

        return {
            "assessment": assessment,
            "confidence": result.confidence,
        }

    async def confidence_node(state: ExtractionState) -> ExtractionState:
        confidence = state.get("confidence", {})
        low_fields = [
            field for field, score in confidence.items() if score < settings.confidence_threshold
        ]

        if not confidence:
            low_fields = ["assessment"]

        return {"low_confidence_fields": low_fields}

    graph = StateGraph(ExtractionState)
    graph.add_node("extract", extract_node)
    graph.add_node("validate", validation_node)
    graph.add_node("confidence", confidence_node)
    graph.set_entry_point("extract")
    graph.add_edge("extract", "validate")
    graph.add_edge("validate", "confidence")
    graph.add_edge("confidence", END)
    return graph.compile()


async def extract_assessment(
    transcript: str,
    settings: Settings,
    extractor: Extractor | None = None,
) -> tuple[FirstAssessment, dict[str, float]]:
    if not has_useful_transcript(transcript):
        raise ExtractionConfidenceError(["transcript"])

    graph = build_extraction_graph(settings, extractor=extractor)
    state = await graph.ainvoke({"transcript": transcript})
    low_fields = state.get("low_confidence_fields", [])

    if low_fields:
        raise ExtractionConfidenceError(low_fields)

    return state["assessment"], state.get("confidence", {})
