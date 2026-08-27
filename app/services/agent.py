"""LangGraph agent that turns a transcript into a validated `FirstAssessment`.

Two nodes: `extract` calls the LLM for structured output, `validate` decides
whether the result clears the confidence threshold. Splitting them keeps the
extraction free of policy and makes the 422 trigger a single, obvious place.
"""

import logging
from typing import List, Optional, TypedDict

from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph

from app.config import CONFIDENCE_THRESHOLD, LLM_MODEL, get_groq_api_key
from app.schemas import ExtractionResult, FirstAssessment
from app.services.prompts import SYSTEM_PROMPT

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    transcript: str
    assessment: Optional[FirstAssessment]
    confidence_score: float
    is_confident: bool
    field_errors: List[str]


def extract_clinical_data_node(state: AgentState) -> dict:
    """Call the Groq LLM to extract a structured `FirstAssessment` and a confidence score.

    This node reports the raw confidence score and any field errors. It does not
    decide whether the result is acceptable - `validate_confidence_node` is the
    single authority on `is_confident`.
    """
    transcript = state.get("transcript", "")
    groq_key = get_groq_api_key()

    if not groq_key:
        logger.error("GROQ_API_KEY environment variable is required.")
        return {
            "assessment": FirstAssessment(),
            "confidence_score": 0.0,
            "field_errors": ["GROQ_API_KEY environment variable is missing."]
        }

    try:
        from langchain_groq import ChatGroq

        logger.info(f"Extracting clinical data using Groq model '{LLM_MODEL}'...")

        llm = ChatGroq(
            model_name=LLM_MODEL,
            temperature=0,
            groq_api_key=groq_key
        )

        structured_llm = llm.with_structured_output(ExtractionResult)

        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", "Clinical Transcript:\n{transcript}")
        ])

        chain = prompt | structured_llm
        result: ExtractionResult = chain.invoke({"transcript": transcript})

        return {
            "assessment": result.assessment,
            "confidence_score": result.confidence_score,
            "field_errors": result.field_errors
        }
    except Exception as e:
        logger.error(f"Error in LLM clinical extraction: {e}")
        return {
            "assessment": FirstAssessment(),
            "confidence_score": 0.0,
            "field_errors": [f"Extraction failed: {str(e)}"]
        }


def validate_confidence_node(state: AgentState) -> dict:
    """Sole authority on whether the extraction is acceptable.

    The spec triggers HTTP 422 on confidence below the threshold, so the gate is
    the score alone. `field_errors` are the field-level detail reported with the
    422, not an additional trigger.
    """
    confidence = state.get("confidence_score", 0.0)
    field_errors = state.get("field_errors", [])

    is_confident = confidence >= CONFIDENCE_THRESHOLD

    if not is_confident and not field_errors:
        field_errors = [
            f"Overall extraction confidence score ({confidence:.2f}) is below "
            f"required threshold ({CONFIDENCE_THRESHOLD})"
        ]

    return {
        "is_confident": is_confident,
        "field_errors": field_errors
    }


def build_clinical_extraction_graph():
    """Construct and compile the LangGraph clinical extraction workflow."""
    workflow = StateGraph(AgentState)

    workflow.add_node("extract", extract_clinical_data_node)
    workflow.add_node("validate", validate_confidence_node)

    workflow.add_edge(START, "extract")
    workflow.add_edge("extract", "validate")
    workflow.add_edge("validate", END)

    return workflow.compile()


# Compiled once at import; the graph is stateless between invocations.
clinical_agent = build_clinical_extraction_graph()


def run_clinical_agent(transcript: str) -> ExtractionResult:
    """Run the compiled pipeline on transcript text.

    The initial state fails closed: if the graph cannot produce a verdict, the
    result is an unconfident empty assessment rather than a confident one.
    """
    initial_state: AgentState = {
        "transcript": transcript,
        "assessment": None,
        "confidence_score": 0.0,
        "is_confident": False,
        "field_errors": []
    }

    final_state = clinical_agent.invoke(initial_state)

    return ExtractionResult(
        assessment=final_state.get("assessment") or FirstAssessment(),
        confidence_score=final_state.get("confidence_score", 0.0),
        is_confident=final_state.get("is_confident", False),
        field_errors=final_state.get("field_errors", [])
    )
