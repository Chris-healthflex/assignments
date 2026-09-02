"""
LangGraph agent: transcript -> FirstAssessment (+ confidence flags).

Graph
-----
    extract  ->  normalize  ->  audit  ->  END
       ^            |
       |  (retry once on schema validation failure)
       +------------+

* extract   : LLM with structured output (ExtractionDraft). Prompt forbids
              inventing clinical values / scores / dates.
* normalize : re-validate into the strict FirstAssessment model so the
              payload is an exact schema match (no extra/renamed keys,
              lists always lists, strings never null).
* audit     : deterministic checks on top of the LLM's self-reported
              confidence — empty core fields, values that don't appear in
              the transcript, etc. Adds FieldFlags and decides low_confidence.
"""
from __future__ import annotations

import logging
import re
from typing import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from pydantic import ValidationError

from .config import settings
from .schemas import ExtractionDraft, ExtractionResult, FieldFlag, FirstAssessment

log = logging.getLogger(__name__)

# Fields whose absence makes the assessment unusable for the frontend.
CORE_FIELDS = (
    "clinicalDetails.chiefComplaint",
    "clinicalDetails.clinicalHistory",
)

SYSTEM_PROMPT = """You are a clinical documentation assistant for a physiotherapy / rehabilitation clinic.
You will be given the raw transcript of a clinician–patient first-assessment session.
Fill the FirstAssessment form STRICTLY from what is said in the transcript.

Hard rules (violations are unacceptable in a medical product):
1. NEVER invent, infer, or "round" clinical values, test scores, ranges of motion, pain scores, dates, or durations.
   If a number or date is not explicitly stated, leave that string EMPTY ("") and add a flag for it.
2. Relative dates ("in six weeks", "by next month") must NOT be converted to a calendar date unless a session
   date is provided below. Otherwise copy the phrase verbatim into the field and flag it.
3. Do not add findings, goals, or advice the clinician did not say. Do not add sections that are not in the schema.
4. Every string must be a string ("" if unknown). Every list must be a list ([] if nothing was said).
5. Use the clinician's wording where possible; light cleanup of speech disfluencies is fine.

Field guidance:
- clinicalDetails.clinicalHistory : relevant history, mechanism of injury, prior treatment, comorbidities.
- clinicalDetails.chiefComplaint  : the main problem in the patient's words / clinician summary.
- clinicalDetails.duration        : how long the complaint has been present (e.g. "3 weeks").
- subjectiveAssessments[]         : patient-reported measures or subjective tests (pain scale, functional questionnaires,
                                    aggravating/easing factors) -> testName + conclusion.
- objectiveAssessment.tests[]     : clinician-measured tests. Put bilateral values in left/right, single values in value.
                                    unitName e.g. "degrees", "kg", "cm", "seconds", "/10". comments for qualitative notes.
- subjectiveGoals[]               : patient-stated goals ("get back to running") + targetDate if stated.
- objectiveGoals[]                : measurable goals -> goalName, goalCategory (e.g. "Range of motion", "Strength",
                                    "Pain", "Function"), unitName, value, targetDate.
- recommendation[]                : treatment plan -> sessionType (e.g. "Physiotherapy", "Manual therapy",
                                    "Home exercise program") + sessionFrequency (e.g. "2x per week for 6 weeks").
- patientAdvice.adviceDetails     : advice / education / home instructions given to the patient.

Confidence:
- `flags`: one entry per field you could not fill confidently, with a dot-path (e.g. "objectiveGoals[0].targetDate"),
  a confidence in [0,1], and the reason.
- `overall_confidence`: your confidence that the filled form faithfully reflects the transcript
  (transcription quality, ambiguity, missing sections all lower it).
"""


class AgentState(TypedDict, total=False):
    transcript: str
    session_date: str | None
    draft: ExtractionDraft
    assessment: FirstAssessment
    flags: list[FieldFlag]
    overall_confidence: float
    low_confidence: bool
    attempts: int
    error: str | None


# --------------------------------------------------------------------------- LLM
def _llm():
    if settings.llm_provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=settings.llm_model, api_key=settings.anthropic_api_key, temperature=0)
    if settings.llm_provider in ("google", "gemini"):
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=settings.llm_model,
            google_api_key=settings.google_api_key,
            temperature=0,
        )
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model=settings.llm_model, api_key=settings.openai_api_key, temperature=0)


# --------------------------------------------------------------------------- nodes
def extract(state: AgentState) -> AgentState:
    structured = _llm().with_structured_output(ExtractionDraft)
    user = f"Transcript:\n\"\"\"\n{state['transcript']}\n\"\"\""
    if state.get("session_date"):
        user = f"Session date: {state['session_date']} (you MAY resolve relative dates against this, ISO YYYY-MM-DD).\n\n" + user
    if state.get("error"):
        user += f"\n\nYour previous answer failed schema validation:\n{state['error']}\nFix it and answer again."
    draft = structured.invoke([SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=user)])
    return {"draft": draft, "attempts": state.get("attempts", 0) + 1, "error": None}


def normalize(state: AgentState) -> AgentState:
    """Re-validate into the strict schema. Guarantees exact-match output."""
    try:
        raw = state["draft"].assessment.model_dump()
        assessment = FirstAssessment.model_validate(raw)
        return {"assessment": assessment, "flags": list(state["draft"].flags),
                "overall_confidence": state["draft"].overall_confidence, "error": None}
    except ValidationError as e:  # pragma: no cover - defensive
        return {"error": str(e)}


def _numbers(s: str) -> set[str]:
    return set(re.findall(r"\d+(?:\.\d+)?", s))


def audit(state: AgentState) -> AgentState:
    """Deterministic guardrails layered on top of the LLM's self-assessment."""
    a = state["assessment"]
    flags = list(state["flags"])
    known = {f.field for f in flags}
    transcript_nums = _numbers(state["transcript"])

    def flag(path: str, conf: float, reason: str):
        if path not in known:
            flags.append(FieldFlag(field=path, confidence=conf, reason=reason))
            known.add(path)

    # 1. Empty core fields
    for path in CORE_FIELDS:
        section, key = path.split(".")
        if not getattr(getattr(a, section), key):
            flag(path, 0.0, "Not found in transcript")

    # 2. Numeric values must literally appear in the transcript (anti-hallucination)
    def check_num(path: str, val: str):
        for n in _numbers(val):
            if n not in transcript_nums:
                flag(path, 0.2, f"Value '{val}' contains a number not present in the transcript")

    for i, t in enumerate(a.objectiveAssessment.tests):
        for k in ("value", "left", "right"):
            check_num(f"objectiveAssessment.tests[{i}].{k}", getattr(t, k))
    for i, g in enumerate(a.objectiveGoals):
        check_num(f"objectiveGoals[{i}].value", g.value)

    # 3. Dates: only allowed if a session date was supplied or the date text is in the transcript
    for i, g in enumerate(a.subjectiveGoals):
        if g.targetDate and not state.get("session_date") and g.targetDate.lower() not in state["transcript"].lower():
            flag(f"subjectiveGoals[{i}].targetDate", 0.3, "Date not stated verbatim and no session date supplied")
    for i, g in enumerate(a.objectiveGoals):
        if g.targetDate and not state.get("session_date") and g.targetDate.lower() not in state["transcript"].lower():
            flag(f"objectiveGoals[{i}].targetDate", 0.3, "Date not stated verbatim and no session date supplied")

    # 4. Decide
    core_low = any(f.field in CORE_FIELDS and f.confidence < settings.confidence_threshold for f in flags)
    low = core_low or state["overall_confidence"] < settings.confidence_threshold
    return {"flags": flags, "low_confidence": low}


def _after_normalize(state: AgentState) -> str:
    if state.get("error") and state.get("attempts", 0) < 2:
        return "extract"
    if state.get("error"):
        raise RuntimeError(f"Extraction failed schema validation twice: {state['error']}")
    return "audit"


# --------------------------------------------------------------------------- graph
def build_graph():
    g = StateGraph(AgentState)
    g.add_node("extract", extract)
    g.add_node("normalize", normalize)
    g.add_node("audit", audit)
    g.set_entry_point("extract")
    g.add_edge("extract", "normalize")
    g.add_conditional_edges("normalize", _after_normalize, {"extract": "extract", "audit": "audit"})
    g.add_edge("audit", END)
    return g.compile()


_GRAPH = None


def run_extraction(transcript: str, session_date: str | None = None) -> ExtractionResult:
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph()
    out = _GRAPH.invoke({"transcript": transcript, "session_date": session_date, "attempts": 0})
    return ExtractionResult(
        assessment=out["assessment"],
        flags=out["flags"],
        overall_confidence=out["overall_confidence"],
        transcript=transcript,
        low_confidence=out["low_confidence"],
    )
