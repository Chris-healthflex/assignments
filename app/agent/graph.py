from langgraph.graph import StateGraph, END

from app.agent.state import AgentState
from app.agent.nodes.clinical_details import extract_clinical_details
from app.agent.nodes.subjective_assessments import extract_subjective_assessments
from app.agent.nodes.objective_assessment import extract_objective_assessment
from app.agent.nodes.goals import extract_goals
from app.agent.nodes.recommendation import extract_recommendation
from app.agent.nodes.patient_advice import extract_patient_advice
from app.agent.nodes.aggregate import aggregate


def build_extraction_graph():
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("extract_clinical_details", extract_clinical_details)
    graph.add_node("extract_subjective_assessments", extract_subjective_assessments)
    graph.add_node("extract_objective_assessment", extract_objective_assessment)
    graph.add_node("extract_goals", extract_goals)
    graph.add_node("extract_recommendation", extract_recommendation)
    graph.add_node("extract_patient_advice", extract_patient_advice)
    graph.add_node("aggregate", aggregate)

    # Entry point
    graph.set_entry_point("extract_clinical_details")

    # Sequential edges
    graph.add_edge("extract_clinical_details", "extract_subjective_assessments")
    graph.add_edge("extract_subjective_assessments", "extract_objective_assessment")
    graph.add_edge("extract_objective_assessment", "extract_goals")
    graph.add_edge("extract_goals", "extract_recommendation")
    graph.add_edge("extract_recommendation", "extract_patient_advice")
    graph.add_edge("extract_patient_advice", "aggregate")

    # Conditional retry
    def should_retry(state: AgentState):
        if state.get("retry_needed", False):
            return "extract_clinical_details"
        return END

    graph.add_conditional_edges(
        "aggregate",
        should_retry,
        {
            "extract_clinical_details": "extract_clinical_details",
            END: END,
        },
    )

    return graph.compile()