"""LangGraph pipeline: clinical transcript -> FirstAssessment.

Two nodes:
  1. extract           - LLM call bound to structured output (ExtractionResult).
  2. check_confidence   - pure Python, decides whether too many sections were
                          flagged as low-confidence by the extraction step.

The LLM is asked to ground every field in the transcript and to name any
top-level FirstAssessment section it could not confidently fill, rather than
inventing clinical values, scores, or dates.
"""

from typing import Protocol, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field
from typing_extensions import NotRequired

from app.schemas.first_assessment import ASSESSMENT_SECTIONS, FirstAssessment

SYSTEM_PROMPT = """You are a clinical documentation assistant. You will be given \
the transcript of a real physiotherapy/clinical session between a clinician and \
a patient.

Extract the information into the FirstAssessment structure using ONLY what is \
explicitly stated or clearly implied in the transcript.

Rules:
- Never invent clinical values, test scores, units, or dates that are not in \
the transcript.
- If a section has no supporting information in the transcript, leave its \
string fields as empty strings and its list fields as empty lists, and add \
that section's name to low_confidence_sections.
- All list fields must be lists (use a single-item list if only one item \
applies, never a bare object).
- Do not add fields beyond the given schema and do not rename keys.
"""


class ExtractionResult(BaseModel):
    assessment: FirstAssessment = Field(default_factory=FirstAssessment)
    low_confidence_sections: list[str] = Field(default_factory=list)


class ExtractionState(TypedDict):
    transcript: str
    result: NotRequired[ExtractionResult]
    is_low_confidence: NotRequired[bool]


class StructuredLLM(Protocol):
    def invoke(self, messages: list) -> ExtractionResult: ...


def _default_llm() -> StructuredLLM:
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model="gpt-4o", temperature=0).with_structured_output(
        ExtractionResult
    )


def _build_extract_node(llm: StructuredLLM):
    def extract(state: ExtractionState) -> dict:
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=state["transcript"]),
        ]
        result = llm.invoke(messages)

        unknown_sections = set(result.low_confidence_sections) - set(
            ASSESSMENT_SECTIONS
        )
        if unknown_sections:
            result.low_confidence_sections = [
                s for s in result.low_confidence_sections if s not in unknown_sections
            ]

        return {"result": result}

    return extract


def _check_confidence(threshold: int):
    def check(state: ExtractionState) -> dict:
        flagged = state["result"].low_confidence_sections
        return {"is_low_confidence": len(flagged) >= threshold}

    return check


def build_extraction_graph(llm: StructuredLLM | None = None, confidence_threshold: int = 2):
    from langgraph.graph import END, StateGraph

    llm = llm or _default_llm()

    graph = StateGraph(ExtractionState)
    graph.add_node("extract", _build_extract_node(llm))
    graph.add_node("check_confidence", _check_confidence(confidence_threshold))
    graph.set_entry_point("extract")
    graph.add_edge("extract", "check_confidence")
    graph.add_edge("check_confidence", END)

    return graph.compile()


def run_extraction(
    transcript: str,
    llm: StructuredLLM | None = None,
    confidence_threshold: int = 2,
) -> tuple[ExtractionResult, bool]:
    """Run the extraction graph and return (result, is_low_confidence)."""
    compiled = build_extraction_graph(llm=llm, confidence_threshold=confidence_threshold)
    final_state = compiled.invoke({"transcript": transcript})
    return final_state["result"], final_state["is_low_confidence"]
