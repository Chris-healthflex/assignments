from __future__ import annotations
import json
import logging
from typing import Any, Dict, List, Optional, Tuple, TypedDict
from pydantic import BaseModel, Field

from app.config import settings
from app.models.assessment import FirstAssessment
from app.models.internal import FieldEvidence, ConfidenceReport
from app.services.grounding import ground_field, is_numeric_field, is_date_or_duration_field
from app.services.confidence import fuse_confidence, evaluate_confidence

logger = logging.getLogger(__name__)


class AgentState(TypedDict, total=False):
    transcript: str
    segments: List[Tuple[float, float, str]]
    session_date: Optional[str]
    draft: Dict[str, Any]
    raw_scores: List[Dict[str, Any]]
    assessment: FirstAssessment
    field_evidence: List[FieldEvidence]
    confidence_report: ConfidenceReport


class RawFieldScore(BaseModel):
    field: str
    llm_confidence: float = 0.90
    reason: str = ""


class ExtractionOutput(BaseModel):
    clinicalDetails: Dict[str, Any] = Field(default_factory=dict)
    subjectiveAssessments: List[Dict[str, Any]] = Field(default_factory=list)
    objectiveAssessment: Dict[str, Any] = Field(default_factory=dict)
    subjectiveGoals: List[Dict[str, Any]] = Field(default_factory=list)
    objectiveGoals: List[Dict[str, Any]] = Field(default_factory=list)
    recommendation: List[Dict[str, Any]] = Field(default_factory=list)
    patientAdvice: Dict[str, Any] = Field(default_factory=dict)
    field_scores: List[RawFieldScore] = Field(default_factory=list)


EXTRACTION_SYSTEM_PROMPT = """You are a specialized clinical data extraction pipeline for Stance Health, a musculoskeletal and sports physiotherapy company.
Your task is to transcribe clinical information from the clinician-patient session transcript into the strict FirstAssessment JSON structure.

NON-NEGOTIABLE SAFETY & CLINICAL ACCURACY RULES:
1. NEVER hallucinate clinical values, ROM degrees, pain scale numbers, dates, or test names.
2. If a value, test, score, date, or recommendation was NOT EXPLICITLY STATED in the transcript, LEAVE THE FIELD AS AN EMPTY STRING ("") OR EMPTY LIST ([]).
3. NEVER round, estimate, or "helpfully" reconstruct a number, score, or date that was not stated.
4. Exactly follow the 7 sections of FirstAssessment:
   - clinicalDetails: {clinicalHistory: str, chiefComplaint: str, duration: str}
   - subjectiveAssessments: list of {testName: str, conclusion: str}
   - objectiveAssessment: {tests: list of {testName: str, unitName: str, value: str, left: str, right: str, comments: str}}
   - subjectiveGoals: list of {goalDetails: str, targetDate: str}
   - objectiveGoals: list of {goalName: str, goalCategory: str, unitName: str, value: str, targetDate: str}
   - recommendation: list of {sessionType: str, sessionFrequency: str}
   - patientAdvice: {adviceDetails: str}
5. For each populated field, report a field_scores item with:
   - field: dot-path to the field (e.g. "clinicalDetails.chiefComplaint", "objectiveAssessment.tests[0].left")
   - llm_confidence: float between 0.0 and 1.0 representing your certainty that this was directly spoken
   - reason: short justification (e.g. "Explicitly stated by clinician")
"""


def get_llm():
    """Initializes the configured LLM provider."""
    provider = (settings.LLM_PROVIDER or "").lower()

    if provider == "gemini":
        if not settings.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY is not set. Please provide it in .env or environment.")
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=settings.GEMINI_MODEL,
            google_api_key=settings.GEMINI_API_KEY,
            temperature=0.0,
        )
    elif provider == "openai":
        if not settings.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is not set. Please provide it in .env or environment.")
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=settings.OPENAI_MODEL,
            api_key=settings.OPENAI_API_KEY,
            temperature=0.0,
        )
    elif provider == "ollama":
        from langchain_community.chat_models import ChatOllama
        return ChatOllama(
            base_url=settings.OLLAMA_BASE_URL,
            model=settings.OLLAMA_MODEL,
            temperature=0.0,
        )
    else:
        # Fallback to OpenAI if key exists, or Gemini if key exists
        if settings.OPENAI_API_KEY:
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(model="gpt-4o-mini", api_key=settings.OPENAI_API_KEY, temperature=0.0)
        if settings.GEMINI_API_KEY:
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=settings.GEMINI_API_KEY, temperature=0.0)
        raise ValueError(f"No configured LLM provider or API key found for '{provider}'.")


def extract_node(state: AgentState) -> Dict[str, Any]:
    """
    Node 1: Structured LLM call to extract clinical data and self-reported field confidence.
    Fails loudly if LLM fails (no rule-based fallback).
    """
    transcript = state.get("transcript", "")
    if not transcript:
        raise ValueError("Cannot extract from empty transcript.")

    llm = get_llm()
    structured_llm = llm.with_structured_output(ExtractionOutput)

    messages = [
        {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
        {"role": "user", "content": f"Clinical Transcript:\n\n{transcript}"},
    ]

    result: ExtractionOutput = structured_llm.invoke(messages)

    draft = {
        "clinicalDetails": result.clinicalDetails or {},
        "subjectiveAssessments": result.subjectiveAssessments or [],
        "objectiveAssessment": result.objectiveAssessment or {"tests": []},
        "subjectiveGoals": result.subjectiveGoals or [],
        "objectiveGoals": result.objectiveGoals or [],
        "recommendation": result.recommendation or [],
        "patientAdvice": result.patientAdvice or {},
    }

    raw_scores = [item.model_dump() for item in result.field_scores]

    return {
        **state,
        "draft": draft,
        "raw_scores": raw_scores,
    }


def ground_node(state: AgentState) -> Dict[str, Any]:
    """
    Node 2: Deterministic transcript check for numeric/date fields and score fusion.
    """
    transcript = state.get("transcript", "")
    draft = state.get("draft", {})
    raw_scores_map = {item["field"]: item for item in state.get("raw_scores", [])}

    field_evidence: List[FieldEvidence] = []

    def check_and_add(path: str, val: Any, anchor: Optional[str] = None):
        val_str = str(val or "").strip()
        if not val_str:
            return  # Empty fields are not flagged

        is_num_or_date = is_numeric_field(path) or is_date_or_duration_field(path)
        grounded, span = ground_field(transcript, path, val_str, anchor=anchor)

        raw_info = raw_scores_map.get(path, {})
        llm_conf = float(raw_info.get("llm_confidence", 0.90))
        reason = raw_info.get("reason", "Directly stated" if grounded else "Ungrounded in transcript")

        fused = fuse_confidence(llm_conf, grounded, is_num_or_date)

        if is_num_or_date and not grounded:
            reason = f"Numeric/date value '{val_str}' not found in transcript near '{anchor or path}'"

        field_evidence.append(
            FieldEvidence(
                field=path,
                confidence=round(fused, 3),
                llm_confidence=round(llm_conf, 3),
                grounded=grounded,
                evidence_span=span,
                reason=reason,
            )
        )

    # 1. clinicalDetails
    cd = draft.get("clinicalDetails", {})
    check_and_add("clinicalDetails.chiefComplaint", cd.get("chiefComplaint"))
    check_and_add("clinicalDetails.duration", cd.get("duration"))
    check_and_add("clinicalDetails.clinicalHistory", cd.get("clinicalHistory"))

    # 2. subjectiveAssessments
    for i, item in enumerate(draft.get("subjectiveAssessments", [])):
        check_and_add(f"subjectiveAssessments[{i}].conclusion", item.get("conclusion"), anchor=item.get("testName"))

    # 3. objectiveAssessment.tests
    oa = draft.get("objectiveAssessment", {})
    for i, t in enumerate(oa.get("tests", [])):
        tname = t.get("testName", "")
        check_and_add(f"objectiveAssessment.tests[{i}].value", t.get("value"), anchor=tname)
        check_and_add(f"objectiveAssessment.tests[{i}].left", t.get("left"), anchor=f"{tname} left")
        check_and_add(f"objectiveAssessment.tests[{i}].right", t.get("right"), anchor=f"{tname} right")

    # 4. subjectiveGoals & objectiveGoals
    for i, sg in enumerate(draft.get("subjectiveGoals", [])):
        check_and_add(f"subjectiveGoals[{i}].targetDate", sg.get("targetDate"), anchor=sg.get("goalDetails"))

    for i, og in enumerate(draft.get("objectiveGoals", [])):
        check_and_add(f"objectiveGoals[{i}].value", og.get("value"), anchor=og.get("goalName"))
        check_and_add(f"objectiveGoals[{i}].targetDate", og.get("targetDate"), anchor=og.get("goalName"))

    # 5. recommendation & patientAdvice
    for i, rec in enumerate(draft.get("recommendation", [])):
        check_and_add(f"recommendation[{i}].sessionFrequency", rec.get("sessionFrequency"), anchor=rec.get("sessionType"))

    pa = draft.get("patientAdvice", {})
    check_and_add("patientAdvice.adviceDetails", pa.get("adviceDetails"))

    return {
        **state,
        "field_evidence": field_evidence,
    }


def normalize_node(state: AgentState) -> Dict[str, Any]:
    """
    Node 3: Strict schema validation through FirstAssessment.
    Coerces nulls, strips whitespace, forbids extra keys.
    """
    draft = state.get("draft", {})
    assessment = FirstAssessment.model_validate(draft)
    return {
        **state,
        "assessment": assessment,
    }


def audit_node(state: AgentState) -> Dict[str, Any]:
    """
    Node 4: Audit score fusion and assemble the final ConfidenceReport.
    """
    field_evidence = state.get("field_evidence", [])
    report = evaluate_confidence(field_evidence, threshold=settings.CONFIDENCE_THRESHOLD)
    return {
        **state,
        "confidence_report": report,
    }


class SequentialPipelineFallback:
    """Fallback safety net executing nodes sequentially if StateGraph.compile() fails."""
    def invoke(self, state: AgentState) -> AgentState:
        s = dict(state)
        s = extract_node(s)
        s = ground_node(s)
        s = normalize_node(s)
        s = audit_node(s)
        return s


def build_pipeline_graph():
    """Builds and compiles the LangGraph StateGraph, with sequential fallback safety net."""
    try:
        from langgraph.graph import StateGraph, END

        builder = StateGraph(AgentState)
        builder.add_node("extract", extract_node)
        builder.add_node("ground", ground_node)
        builder.add_node("normalize", normalize_node)
        builder.add_node("audit", audit_node)

        builder.set_entry_point("extract")
        builder.add_edge("extract", "ground")
        builder.add_edge("ground", "normalize")
        builder.add_edge("normalize", "audit")
        builder.add_edge("audit", END)

        return builder.compile()
    except Exception as exc:
        logger.warning("Failed to compile LangGraph StateGraph (%s), using sequential fallback.", exc)
        return SequentialPipelineFallback()
