"""LangGraph pipeline: clinical transcript -> grounded FirstAssessment.

Four nodes, with a correction loop:

    extract ──▶ validate ──▶ refine ──┐
                   │  ▲               │
                   │  └───────────────┘
                   ▼
             check_confidence ──▶ END

`extract` asks the LLM for the assessment *plus* the transcript segments each
value came from. `validate` is pure Python — it re-reads the model's answer
against the transcript and catches the failure modes a prompt alone doesn't
prevent: values with no supporting segment, placeholder junk like "N/A",
citations pointing at segments that don't exist, and sections claimed as
confident while sitting empty. If it finds any, `refine` sends the model its
own output plus the specific complaints and asks for a correction.

The grounding check is what gives "never hallucinate" real teeth: a field the
model cannot point to in the transcript is reported as ungrounded rather than
quietly trusted.
"""

from __future__ import annotations

import logging
from typing import Protocol, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field
from typing_extensions import NotRequired

from app.schemas.first_assessment import ASSESSMENT_SECTIONS, FirstAssessment
from app.services.transcription import Transcript

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "openai/gpt-oss-120b"
MAX_REFINEMENTS = 1

# Values that mean "the model had nothing" but were written as text anyway.
# These must become empty strings, otherwise they leak into the clinician's
# form as if they were real findings.
PLACEHOLDER_VALUES = {
    "n/a",
    "na",
    "none",
    "null",
    "nil",
    "unknown",
    "not mentioned",
    "not stated",
    "not specified",
    "not provided",
    "not discussed",
    "not applicable",
    "no data",
    "tbd",
    "-",
    "--",
    "?",
}

SYSTEM_PROMPT = """You are a clinical documentation assistant. You will be given \
the transcript of a real physiotherapy/clinical session between a clinician and \
a patient.

The transcript is presented as numbered, time-coded segments:
    [12] (84.0s-91.5s) and how long has the knee been bothering you?

Extract the information into the FirstAssessment structure using ONLY what is \
explicitly stated or clearly implied in the transcript.

Rules:
- Never invent clinical values, test scores, units, or dates that are not in \
the transcript.
- For EVERY field you fill, add an entry to `evidence` naming the field path \
and the segment id(s) that support it, with a short verbatim quote. Field \
paths use dots and indices, e.g. "clinicalDetails.chiefComplaint", \
"subjectiveAssessments[0].testName", "objectiveAssessment.tests[1].value".
- If you cannot cite a segment for a value, do not write the value.
- If a section has no supporting information in the transcript, leave its \
string fields as empty strings and its list fields as empty lists, and add \
that section's name to low_confidence_sections.
- Never write placeholder text such as "N/A", "unknown", "not mentioned" or \
"TBD". Use an empty string instead.
- All list fields must be lists (use a single-item list if only one item \
applies, never a bare object).
- Do not add fields beyond the given schema and do not rename keys.
"""

REFINE_PROMPT = """Your previous extraction was checked against the transcript \
and the following problems were found:

{issues}

Here is your previous answer:

{previous}

Produce a corrected extraction of the SAME transcript. Fix only the problems \
listed above and leave everything else unchanged. Remember: if a value cannot \
be cited to a transcript segment, remove the value rather than keeping it.
"""


class FieldEvidence(BaseModel):
    """Where in the recording a single extracted value came from."""

    field: str = Field(description="Dotted path of the field this supports")
    segmentIds: list[int] = Field(
        default_factory=list, description="Transcript segment ids supporting the value"
    )
    quote: str = Field(default="", description="Short verbatim supporting quote")


class ExtractionResult(BaseModel):
    """The LLM's structured output."""

    assessment: FirstAssessment = Field(default_factory=FirstAssessment)
    low_confidence_sections: list[str] = Field(default_factory=list)
    evidence: list[FieldEvidence] = Field(default_factory=list)


class ExtractionReport(BaseModel):
    """What the pipeline returns: the model's answer plus our audit of it."""

    assessment: FirstAssessment = Field(default_factory=FirstAssessment)
    low_confidence_sections: list[str] = Field(default_factory=list)
    evidence: list[FieldEvidence] = Field(default_factory=list)
    ungrounded_fields: list[str] = Field(default_factory=list)
    validation_issues: list[str] = Field(default_factory=list)
    attempts: int = 1


class ExtractionState(TypedDict):
    transcript: str
    segment_ids: list[int]
    check_grounding: bool
    result: NotRequired[ExtractionResult]
    issues: NotRequired[list[str]]
    ungrounded: NotRequired[list[str]]
    attempts: NotRequired[int]
    is_low_confidence: NotRequired[bool]


class StructuredLLM(Protocol):
    def invoke(self, messages: list) -> ExtractionResult: ...


def _default_llm(api_key: str | None = None) -> StructuredLLM:
    from langchain_groq import ChatGroq

    kwargs = {"model": DEFAULT_MODEL, "temperature": 0}
    if api_key:
        kwargs["api_key"] = api_key

    return ChatGroq(**kwargs).with_structured_output(ExtractionResult)


# --------------------------------------------------------------------------
# Validation helpers (pure functions — testable without an LLM)
# --------------------------------------------------------------------------


def populated_paths(assessment: FirstAssessment) -> list[str]:
    """Every leaf field the model actually filled in, as a dotted path."""
    paths: list[str] = []

    def walk(value, prefix: str) -> None:
        if isinstance(value, BaseModel):
            for name in type(value).model_fields:
                walk(getattr(value, name), f"{prefix}.{name}" if prefix else name)
        elif isinstance(value, list):
            for index, item in enumerate(value):
                walk(item, f"{prefix}[{index}]")
        elif isinstance(value, str) and value.strip():
            paths.append(prefix)

    walk(assessment, "")
    return paths


def _is_grounded(path: str, evidence_fields: set[str]) -> bool:
    """True if some evidence entry covers this path.

    Accepts an exact match or a parent path, so a citation on
    `subjectiveAssessments[0]` grounds `subjectiveAssessments[0].testName`.
    """
    if path in evidence_fields:
        return True
    return any(
        path.startswith(field) and path[len(field)] in ".["
        for field in evidence_fields
        if field and len(path) > len(field)
    )


def _placeholder_paths(assessment: FirstAssessment) -> list[str]:
    found: list[str] = []

    def walk(value, prefix: str) -> None:
        if isinstance(value, BaseModel):
            for name in type(value).model_fields:
                walk(getattr(value, name), f"{prefix}.{name}" if prefix else name)
        elif isinstance(value, list):
            for index, item in enumerate(value):
                walk(item, f"{prefix}[{index}]")
        elif isinstance(value, str) and value.strip().lower() in PLACEHOLDER_VALUES:
            found.append(prefix)

    walk(assessment, "")
    return found


def _section_is_empty(assessment: FirstAssessment, section: str) -> bool:
    return not any(
        path == section or path.startswith((f"{section}.", f"{section}["))
        for path in populated_paths(assessment)
    )


def validate_extraction(
    result: ExtractionResult,
    segment_ids: set[int],
    check_grounding: bool,
) -> tuple[list[str], list[str]]:
    """Audit an extraction. Returns (issues, ungrounded_field_paths)."""
    issues: list[str] = []
    filled = populated_paths(result.assessment)

    placeholders = _placeholder_paths(result.assessment)
    if placeholders:
        issues.append(
            "These fields contain placeholder text instead of an empty string: "
            + ", ".join(sorted(placeholders))
        )

    # A section can't be both confidently skipped and full of data.
    contradictions = [
        section
        for section in result.low_confidence_sections
        if not _section_is_empty(result.assessment, section)
    ]
    if contradictions:
        issues.append(
            "These sections are listed in low_confidence_sections but contain "
            "extracted data — either remove the data or remove the flag: "
            + ", ".join(sorted(contradictions))
        )

    if not filled and not result.low_confidence_sections:
        issues.append(
            "The assessment is entirely empty but no sections were flagged in "
            "low_confidence_sections. Flag the sections you could not fill."
        )

    ungrounded: list[str] = []
    if check_grounding:
        bad_ids = sorted(
            {
                seg_id
                for entry in result.evidence
                for seg_id in entry.segmentIds
                if seg_id not in segment_ids
            }
        )
        if bad_ids:
            issues.append(
                "Evidence cites transcript segments that do not exist: "
                + ", ".join(str(i) for i in bad_ids)
            )

        evidence_fields = {e.field.strip() for e in result.evidence if e.field.strip()}
        ungrounded = [p for p in filled if not _is_grounded(p, evidence_fields)]
        if ungrounded:
            issues.append(
                "These fields have a value but no supporting transcript segment. "
                "Either cite the segment that supports each one, or clear the "
                "value: " + ", ".join(sorted(ungrounded))
            )

    return issues, ungrounded


def _sanitize(result: ExtractionResult, segment_ids: set[int]) -> ExtractionResult:
    """Drop things we can fix ourselves rather than spend an LLM round-trip on."""
    valid_sections = set(ASSESSMENT_SECTIONS)
    result.low_confidence_sections = [
        section
        for section in dict.fromkeys(result.low_confidence_sections)
        if section in valid_sections
    ]

    if segment_ids:
        for entry in result.evidence:
            entry.segmentIds = [i for i in entry.segmentIds if i in segment_ids]

    return result


# --------------------------------------------------------------------------
# Graph nodes
# --------------------------------------------------------------------------


def _build_extract_node(llm: StructuredLLM):
    def extract(state: ExtractionState) -> dict:
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=state["transcript"]),
        ]
        result = _sanitize(llm.invoke(messages), set(state["segment_ids"]))
        return {"result": result, "attempts": 1}

    return extract


def _build_validate_node():
    def validate(state: ExtractionState) -> dict:
        issues, ungrounded = validate_extraction(
            state["result"],
            set(state["segment_ids"]),
            state["check_grounding"],
        )
        if issues:
            logger.info("extraction validation found %d issue(s)", len(issues))
        return {"issues": issues, "ungrounded": ungrounded}

    return validate


def _build_refine_node(llm: StructuredLLM):
    def refine(state: ExtractionState) -> dict:
        previous = state["result"].model_dump_json(indent=2)
        feedback = REFINE_PROMPT.format(
            issues="\n".join(f"- {issue}" for issue in state["issues"]),
            previous=previous,
        )
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=state["transcript"]),
            HumanMessage(content=feedback),
        ]
        result = _sanitize(llm.invoke(messages), set(state["segment_ids"]))
        return {"result": result, "attempts": state.get("attempts", 1) + 1}

    return refine


def _build_confidence_node(threshold: int):
    def check_confidence(state: ExtractionState) -> dict:
        flagged = state["result"].low_confidence_sections
        return {"is_low_confidence": len(flagged) >= threshold}

    return check_confidence


def _build_router(max_refinements: int):
    def route(state: ExtractionState) -> str:
        if state.get("issues") and state.get("attempts", 1) <= max_refinements:
            return "refine"
        return "check_confidence"

    return route


def build_extraction_graph(
    llm: StructuredLLM | None = None,
    confidence_threshold: int = 2,
    api_key: str | None = None,
    max_refinements: int = MAX_REFINEMENTS,
):
    from langgraph.graph import END, StateGraph

    llm = llm or _default_llm(api_key=api_key)

    graph = StateGraph(ExtractionState)
    graph.add_node("extract", _build_extract_node(llm))
    graph.add_node("validate", _build_validate_node())
    graph.add_node("refine", _build_refine_node(llm))
    graph.add_node("check_confidence", _build_confidence_node(confidence_threshold))

    graph.set_entry_point("extract")
    graph.add_edge("extract", "validate")
    graph.add_conditional_edges(
        "validate",
        _build_router(max_refinements),
        {"refine": "refine", "check_confidence": "check_confidence"},
    )
    graph.add_edge("refine", "validate")
    graph.add_edge("check_confidence", END)

    return graph.compile()


def run_extraction(
    transcript: str | Transcript,
    llm: StructuredLLM | None = None,
    confidence_threshold: int = 2,
    api_key: str | None = None,
    max_refinements: int = MAX_REFINEMENTS,
) -> tuple[ExtractionReport, bool]:
    """Run the extraction graph and return (report, is_low_confidence).

    Accepts either a `Transcript` (time-coded — enables evidence grounding) or
    a plain string (no segments to cite, so grounding checks are skipped).
    """
    if isinstance(transcript, Transcript):
        prompt_text = transcript.as_prompt()
        segment_ids = [segment.id for segment in transcript.segments]
    else:
        prompt_text = transcript
        segment_ids = []

    compiled = build_extraction_graph(
        llm=llm,
        confidence_threshold=confidence_threshold,
        api_key=api_key,
        max_refinements=max_refinements,
    )

    final_state = compiled.invoke(
        {
            "transcript": prompt_text,
            "segment_ids": segment_ids,
            "check_grounding": bool(segment_ids),
        }
    )

    result: ExtractionResult = final_state["result"]
    report = ExtractionReport(
        assessment=result.assessment,
        low_confidence_sections=result.low_confidence_sections,
        evidence=result.evidence,
        ungrounded_fields=final_state.get("ungrounded", []),
        validation_issues=final_state.get("issues", []),
        attempts=final_state.get("attempts", 1),
    )

    return report, final_state["is_low_confidence"]
