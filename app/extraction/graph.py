"""The LangGraph extraction agent.

Fans the transcript out across focused section nodes, each of which calls the
configured extractor (Ollama or the deterministic stub). Nodes are independent, so
one section failing/omitting does not corrupt the others. If `langgraph` is not
installed the same nodes run through a tiny sequential fallback runner — identical
outputs, zero behavioural difference — so the pipeline never hard-depends on the lib.
"""
from __future__ import annotations

import time
from typing import Callable, Dict

from app.extraction.llm import get_extractor
from app.extraction.state import ExtractionState


def _timed(state: ExtractionState, key: str, fn: Callable[[], None]) -> None:
    t0 = time.perf_counter()
    fn()
    state.setdefault("timings", {})[key] = round(time.perf_counter() - t0, 2)


def _node_clinical(state: ExtractionState) -> ExtractionState:
    ex = get_extractor()

    def run() -> None:
        state["clinicalDetails"] = ex.extract("clinicalDetails", state["transcript"])

    _timed(state, "clinicalDetails", run)
    return state


def _node_subjective(state: ExtractionState) -> ExtractionState:
    ex = get_extractor()

    def run() -> None:
        out = ex.extract("subjective", state["transcript"])
        state["subjectiveAssessments"] = out.get("items", []) if isinstance(out, dict) else []

    _timed(state, "subjective", run)
    return state


def _node_objective(state: ExtractionState) -> ExtractionState:
    ex = get_extractor()

    def run() -> None:
        out = ex.extract("objective", state["transcript"])
        state["objectiveTests"] = out.get("tests", []) if isinstance(out, dict) else []

    _timed(state, "objective", run)
    return state


def _node_goals(state: ExtractionState) -> ExtractionState:
    ex = get_extractor()

    def run() -> None:
        out = ex.extract("goals", state["transcript"])
        if isinstance(out, dict):
            state["subjectiveGoals"] = out.get("subjectiveGoals", [])
            state["objectiveGoals"] = out.get("objectiveGoals", [])
        else:
            state["subjectiveGoals"] = []
            state["objectiveGoals"] = []

    _timed(state, "goals", run)
    return state


def _node_plan(state: ExtractionState) -> ExtractionState:
    ex = get_extractor()

    def run() -> None:
        out = ex.extract("plan", state["transcript"])
        if isinstance(out, dict):
            state["recommendation"] = out.get("recommendation", [])
            state["patientAdvice"] = out.get("patientAdvice", {})
        else:
            state["recommendation"] = []
            state["patientAdvice"] = {}

    _timed(state, "plan", run)
    return state


_NODES: Dict[str, Callable[[ExtractionState], ExtractionState]] = {
    "clinical": _node_clinical,
    "subjective": _node_subjective,
    "objective": _node_objective,
    "goals": _node_goals,
    "plan": _node_plan,
}


def _build_langgraph():
    """Compile a real StateGraph if langgraph is available, else return None."""
    try:
        from langgraph.graph import END, START, StateGraph
    except ImportError:
        return None

    g = StateGraph(ExtractionState)
    g.add_node("clinical", _node_clinical)
    g.add_node("subjective", _node_subjective)
    g.add_node("objective", _node_objective)
    g.add_node("goals", _node_goals)
    g.add_node("plan", _node_plan)
    g.add_edge(START, "clinical")
    g.add_edge("clinical", "subjective")
    g.add_edge("subjective", "objective")
    g.add_edge("objective", "goals")
    g.add_edge("goals", "plan")
    g.add_edge("plan", END)
    return g.compile()


_COMPILED = _build_langgraph()


def run_extraction(transcript: str) -> ExtractionState:
    """Run all section nodes over the transcript and return the populated state."""
    state: ExtractionState = {"transcript": transcript, "timings": {}}
    if _COMPILED is not None:
        return _COMPILED.invoke(state)
    # Sequential fallback (no langgraph installed): same nodes, same order.
    for node in _NODES.values():
        state = node(state)
    return state
