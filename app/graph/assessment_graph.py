from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from app.models.assessment import FirstAssessment
from app.services.extraction import (
    ExtractionConfidence,
    extract_assessment,
    verify_assessment_grounding,
)


class AssessmentState(TypedDict, total=False):
    transcript: str
    assessment: FirstAssessment
    confidence: ExtractionConfidence


def extract_node(state: AssessmentState) -> AssessmentState:
    transcript = state["transcript"]

    assessment = extract_assessment(transcript)

    return {
        **state,
        "assessment": assessment,
    }


def confidence_node(state: AssessmentState) -> AssessmentState:
    transcript = state["transcript"]
    assessment = state["assessment"]

    confidence = verify_assessment_grounding(
        transcript=transcript,
        assessment=assessment,
    )

    return {
        **state,
        "confidence": confidence,
    }


def build_assessment_graph():
    graph = StateGraph(AssessmentState)

    graph.add_node(
        "extract_assessment",
        extract_node,
    )

    graph.add_node(
        "verify_grounding",
        confidence_node,
    )

    graph.add_edge(
        START,
        "extract_assessment",
    )

    graph.add_edge(
        "extract_assessment",
        "verify_grounding",
    )

    graph.add_edge(
        "verify_grounding",
        END,
    )

    return graph.compile()