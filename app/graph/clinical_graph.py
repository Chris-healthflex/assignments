import os
from typing import TypedDict

from langgraph.graph import StateGraph, START, END

from app.models.assessment import FirstAssessment
from app.models.confidence import ConfidenceReport
from app.services.extraction import extract_assessment_with_confidence


class ClinicalState(TypedDict, total=False):
    transcript: str
    assessment: FirstAssessment
    confidence: ConfidenceReport
    error: dict


def extract_node(state: ClinicalState) -> ClinicalState:
    transcript = state.get("transcript", "").strip()

    if not transcript:
        return {
            "error": {
                "message": "Transcript is empty.",
                "fields": ["transcript"],
            }
        }

    try:
        assessment, confidence = extract_assessment_with_confidence(transcript)

        threshold = float(
            os.getenv("CONFIDENCE_THRESHOLD", "0.70")
        )

        confidence_data = confidence.model_dump()

        low_confidence_fields = {
            field: score
            for field, score in confidence_data.items()
            if score < threshold
        }

        if low_confidence_fields:
            return {
                "error": {
                    "message": "Extraction confidence is below the required threshold.",
                    "threshold": threshold,
                    "fields": [
                        {
                            "field": field,
                            "confidence": score,
                            "message": (
                                f"Confidence {score:.2f} is below "
                                f"the required threshold {threshold:.2f}."
                            ),
                        }
                        for field, score in low_confidence_fields.items()
                    ],
                }
            }

        return {
            "assessment": assessment,
            "confidence": confidence,
        }

    except Exception as exc:
        return {
            "error": {
                "message": "Clinical assessment extraction failed.",
                "error": str(exc),
            }
        }


def build_clinical_graph():
    graph = StateGraph(ClinicalState)

    graph.add_node(
        "extract_clinical_assessment",
        extract_node,
    )

    graph.add_edge(
        START,
        "extract_clinical_assessment",
    )

    graph.add_edge(
        "extract_clinical_assessment",
        END,
    )

    return graph.compile()


clinical_graph = build_clinical_graph()