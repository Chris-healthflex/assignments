"""LangGraph agent that extracts an assessment and repairs what it cannot verify.

The graph exists for its cycle: ``verify`` is deterministic and feeds a
conditional edge back into ``repair`` when fields fail grounding. Without that
loop this would be a linear chain and a graph library would be ornamental.

    START -> extract -> verify -> ungrounded and attempts left -> repair -> verify
                              \\-> otherwise -------------------> finalize -> END
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from .extraction_models import RawExtraction
from .grounding import FieldVerdict, GroundingReport, score
from .mapping import to_first_assessment
from .prompts import build_extraction_prompt, build_repair_prompt
from .schema import FirstAssessment

logger = logging.getLogger(__name__)

# Two repair rounds. Each costs a full LLM call, and a model that has failed the
# same field twice is not going to succeed on a third identical request.
MAX_REPAIR_ATTEMPTS = 2

# Takes a fully built user prompt, returns the model's extraction. Injected so
# the graph can be driven by a stub in tests and by Anthropic in production.
ExtractionRequest = Callable[[str], RawExtraction]


class AssessmentState(TypedDict):
    """State carried between graph nodes.

    Attributes:
        transcript: The session transcript being extracted from.
        extraction: The latest evidence-carrying extraction, if any.
        report: Verdicts and confidence for that extraction, if verified.
        assessment: The production assessment, set only by ``finalize``.
        attempts: How many repair rounds have run.
    """

    transcript: str
    extraction: RawExtraction | None
    report: GroundingReport | None
    assessment: FirstAssessment | None
    attempts: int


@dataclass(frozen=True)
class AssessmentOutcome:
    """What the agent produces for one transcript.

    Attributes:
        assessment: The assessment, with unverifiable fields blanked.
        confidence: Grounded fields divided by total fields.
        ungrounded_fields: The verdicts that never passed verification.
        attempts: How many repair rounds were needed.
    """

    assessment: FirstAssessment
    confidence: float
    ungrounded_fields: tuple[FieldVerdict, ...]
    attempts: int


def build_agent(request: ExtractionRequest) -> CompiledStateGraph:
    """Compile the extraction graph.

    Args:
        request: Callable that turns a prompt into an extraction.

    Returns:
        The compiled graph, invoked with ``.invoke(initial_state)``.
    """
    graph: StateGraph = StateGraph(AssessmentState)
    graph.add_node("extract", partial(_extract, request=request))
    graph.add_node("verify", _verify)
    graph.add_node("repair", partial(_repair, request=request))
    graph.add_node("finalize", _finalize)

    graph.add_edge(START, "extract")
    graph.add_edge("extract", "verify")
    graph.add_conditional_edges(
        "verify", _route_after_verify, {"repair": "repair", "finalize": "finalize"}
    )
    graph.add_edge("repair", "verify")
    graph.add_edge("finalize", END)
    return graph.compile()


def run_agent(transcript: str, request: ExtractionRequest) -> AssessmentOutcome:
    """Run the extraction graph over one transcript.

    Never raises on low confidence: the confidence gate belongs to the API
    layer, which needs the assessment and the verdicts to build its response.

    Args:
        transcript: The session transcript.
        request: Callable that turns a prompt into an extraction.

    Returns:
        The assessment together with its confidence and failed fields.

    Raises:
        ExtractionError: Propagated from the underlying model call.
    """
    initial_state: AssessmentState = {
        "transcript": transcript,
        "extraction": None,
        "report": None,
        "assessment": None,
        "attempts": 0,
    }
    final_state = build_agent(request).invoke(initial_state)

    report = final_state["report"]
    return AssessmentOutcome(
        assessment=final_state["assessment"],
        confidence=report.confidence,
        ungrounded_fields=report.ungrounded,
        attempts=final_state["attempts"],
    )


def _extract(state: AssessmentState, request: ExtractionRequest) -> dict[str, object]:
    """First pass over the full transcript."""
    prompt = build_extraction_prompt(state["transcript"])
    return {"extraction": request(prompt)}


def _verify(state: AssessmentState) -> dict[str, object]:
    """Check the extraction against the transcript. Deterministic, no model."""
    extraction = state["extraction"]
    assert extraction is not None, "verify runs only after extract or repair"
    return {"report": score(extraction, state["transcript"])}


def _repair(state: AssessmentState, request: ExtractionRequest) -> dict[str, object]:
    """Re-ask for the ungrounded fields only, naming the rejected quotes."""
    report = state["report"]
    assert report is not None, "repair runs only after verify"

    # Field paths only: the rejected quotes are clinical material.
    logger.info(
        "Repair round %d for %d ungrounded field(s)",
        state["attempts"] + 1,
        len(report.ungrounded),
    )
    prompt = build_repair_prompt(state["transcript"], report.rejected_quotes())
    return {"extraction": request(prompt), "attempts": state["attempts"] + 1}


def _finalize(state: AssessmentState) -> dict[str, object]:
    """Map the verified extraction onto the production schema."""
    extraction = state["extraction"]
    report = state["report"]
    assert extraction is not None and report is not None, "finalize runs last"
    return {"assessment": to_first_assessment(extraction, report)}


def _route_after_verify(state: AssessmentState) -> str:
    """Choose between another repair round and finalising.

    The attempts cap is what makes the cycle terminate: a model that keeps
    failing the same fields cannot spin the graph indefinitely.
    """
    report = state["report"]
    assert report is not None, "routing runs only after verify"
    if report.ungrounded and state["attempts"] < MAX_REPAIR_ATTEMPTS:
        return "repair"
    return "finalize"
