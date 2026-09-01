from typing import TypedDict

from langgraph.graph import StateGraph, END

from app.extractor import extract_clinical_assessment


class AssessmentState(TypedDict):
    transcription: str
    assessment: object


def extract_node(state: AssessmentState):
    assessment = extract_clinical_assessment(
        state["transcription"]
    )

    return {
        "assessment": assessment
    }


builder = StateGraph(AssessmentState)

builder.add_node(
    "extract_clinical_assessment",
    extract_node
)

builder.set_entry_point(
    "extract_clinical_assessment"
)

builder.add_edge(
    "extract_clinical_assessment",
    END
)

clinical_graph = builder.compile()