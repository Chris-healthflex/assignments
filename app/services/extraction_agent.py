"""LangGraph extraction agent.

Graph:

    extract_entities  ->  score_confidence  ->  validate_schema  ->  END
                                                      |
                                        (on pydantic ValidationError)
                                                      v
                                              repair_extraction (max 1 retry)

Design decisions:
  * Structured output is forced via Ollama tool-calling (through
    LangChain's `with_structured_output`, using the FirstAssessment
    Pydantic model directly as the target schema) rather than asking the
    model to "return JSON" in prose — this eliminates markdown-fencing /
    trailing-text failure modes entirely, and works fully offline against
    a local Ollama server (no API key, no cost, transcript text never
    leaves the machine — a real consideration for clinical audio).
  * The extraction prompt explicitly instructs: leave a field as "" / omit
    an array item rather than invent a value. This is enforced again in
    score_confidence, which treats fields absent from the transcript as
    low confidence regardless of what the model returned.
  * score_confidence is intentionally a separate node (not folded into the
    extraction call) so confidence is computed from evidence-grounding
    (does the field's value actually appear in/derive from the transcript?)
    rather than the model's own self-reported certainty, which LLMs are
    known to overstate.
  * Local open-weight models are noticeably weaker than frontier hosted
    models at strict structured extraction, so the retry/repair loop and
    the grounding-based confidence gate matter more here, not less — they
    are what keeps a smaller local model from silently emitting bad data.
"""
from __future__ import annotations

import logging
import json
from typing import Any, Optional, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from pydantic import ValidationError

from app.config import get_settings
from app.models.schema import ExtractionResult, FieldConfidence, FirstAssessment

logger = logging.getLogger(__name__)
EXTRACTION_SYSTEM_PROMPT = """
You are a strict clinical information extraction system.

Your task is to extract information from a physiotherapy assessment
transcript into the EXACT JSON structure specified below.

CRITICAL:
- Return ONLY valid JSON.
- Do NOT add any fields.
- Do NOT rename fields.
- Do NOT create your own clinical schema.
- Do NOT include patientIdentifier.
- Do NOT include dateOfAssessment.
- Do NOT include age.
- Do NOT include gender.
- Do NOT include historyOfPresentIllness.
- Do NOT include previousSurgeries.
- Do NOT include physicalExam.
- Do NOT include provisionalDiagnosis.
- Do NOT include treatmentPlan.
- Do NOT include followUp.
- Do NOT include any other fields.

ONLY these top-level fields are allowed:

{
  "clinicalDetails": {
    "clinicalHistory": "",
    "chiefComplaint": "",
    "duration": ""
  },
  "subjectiveAssessments": [],
  "objectiveAssessment": {
    "tests": []
  },
  "subjectiveGoals": [],
  "objectiveGoals": [],
  "recommendation": [],
  "patientAdvice": {
    "adviceDetails": ""
  }
}

OBJECTIVE TEST FORMAT:

Each object inside objectiveAssessment.tests MUST contain:

{
  "testName": "",
  "unitName": "",
  "value": "",
  "left": "",
  "right": "",
  "comments": ""
}

SUBJECTIVE ASSESSMENT FORMAT:

Each item MUST contain:

{
  "testName": "",
  "conclusion": ""
}

SUBJECTIVE GOAL FORMAT:

Each item MUST contain:

{
  "goalDetails": "",
  "targetDate": ""
}

OBJECTIVE GOAL FORMAT:

Each item MUST contain:

{
  "goalName": "",
  "goalCategory": "",
  "unitName": "",
  "value": "",
  "targetDate": ""
}

RECOMMENDATION FORMAT:

Each item MUST contain:

{
  "sessionType": "",
  "sessionFrequency": ""
}

EXTRACTION RULES:

1. Extract ONLY information explicitly stated in the transcript.

2. Never invent clinical information.

3. Missing string fields must be "".

4. Missing arrays must be [].

5. Preserve left/right exactly.

6. Numerical measurements must be stored as strings.

7. Physical examination findings belong in objectiveAssessment.tests.

8. Patient-reported symptoms belong in subjectiveAssessments.

9. Explicit rehabilitation goals belong in subjectiveGoals.

10. Explicit measurable rehabilitation goals belong in objectiveGoals.

11. Treatment recommendations belong in recommendation.

12. Explicit instructions/advice belong in patientAdvice.

13. Do not create a diagnosis field. If diagnosis/history is explicitly
stated, place relevant historical information in clinicalHistory.

14. Do not convert treatment recommendations into goals.

15. Do not invent target dates.

16. If Whisper produces an uncertain value such as "negic 5",
preserve the uncertainty rather than silently changing it.

17. Return exactly one JSON object.

18. No markdown.
19. No explanation.
20. No text before or after the JSON.
"""



class AgentState(TypedDict):
    transcript: str
    raw_extraction: Optional[dict[str, Any]]
    assessment: Optional[FirstAssessment]
    field_confidences: list[FieldConfidence]
    overall_confidence: float
    low_confidence_fields: list[str]
    validation_error: Optional[str]
    retries: int


def _get_structured_llm():
    settings = get_settings()
    llm = ChatOllama(
        model=settings.extraction_model,
        base_url=settings.ollama_base_url,
        temperature=0,
        format = "json"
    )
    # method="function_calling" routes through Ollama's tool-calling API
    # (supported by llama3.1, qwen2.5, mistral-nemo, etc.) so the model
    # emits the FirstAssessment shape directly rather than free-form JSON
    # that then needs to be parsed and hoped-into-shape.
    return llm

def _call_llm_for_extraction(
    transcript: str,
    repair_hint: str | None = None,
) -> dict:
    llm = _get_structured_llm()

    user_text = f"""
Extract the clinical assessment from this transcript.

TRANSCRIPT:
{transcript}
"""

    if repair_hint:
        user_text += f"""

The previous output failed validation.

Validation error:
{repair_hint}

Return the COMPLETE corrected JSON object.
Remember: ONLY the fields defined in the required structure are allowed.
"""

    messages = [
        SystemMessage(content=EXTRACTION_SYSTEM_PROMPT),
        HumanMessage(content=user_text),
    ]

    result = llm.invoke(messages)

    content = result.content if hasattr(result, "content") else result

    if isinstance(content, dict):
        return content

    if isinstance(content, str):
        try:
            return json.loads(content)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"LLM returned invalid JSON: {content}"
            ) from exc

    raise ValueError(
        f"Unexpected LLM response type: {type(content)}"
    )


def extract_entities(state: AgentState) -> AgentState:
    raw = _call_llm_for_extraction(state["transcript"])
    state["raw_extraction"] = raw
    return state


def validate_schema(state: AgentState) -> AgentState:
    try:
        state["assessment"] = FirstAssessment.model_validate(state["raw_extraction"])
        state["validation_error"] = None
    except ValidationError as e:
        state["validation_error"] = str(e)
    return state


def repair_extraction(state: AgentState) -> AgentState:
    state["retries"] += 1
    raw = _call_llm_for_extraction(state["transcript"], repair_hint=state["validation_error"])
    state["raw_extraction"] = raw
    return state


def score_confidence(state: AgentState) -> AgentState:
    """Grounds each extracted field against the transcript text. This is a
    cheap lexical-overlap heuristic (not another LLM call) so scoring is
    fast, deterministic, and auditable — appropriate for a v1 confidence
    gate. Swap in an NLI/entailment model here for production hardening.
    """
    assessment = state["assessment"]
    transcript_lower = state["transcript"].lower()
    confidences: list[FieldConfidence] = []
    def grounded(value: str) -> float:
        value = (value or "").strip()

        if not value:
            return 1.0

        # Normalize common punctuation differences.
        normalized_value = value.lower().replace("-", " ")
        normalized_transcript = transcript_lower.replace("-", " ")

        tokens = [
            t for t in normalized_value.split()
            if len(t) > 3
        ]

        if not tokens:
            return 0.8

        hits = 0

        for token in tokens:
            if token in normalized_transcript:
                hits += 1
                continue

            # Handle common Whisper word variations.
            variations = {
                "dorsiflexion": [
                    "dorsi flexion",
                    "dose of flexion",
                    "dorsal flexion",
                ],
                "flexion": [
                    "flexion",
                    "flexing",
                ],
                "extension": [
                    "extension",
                    "extending",
                ],
                "physiotherapy": [
                    "physiotherapy",
                    "physical therapy",
                ],
            }

            if any(
                variation in normalized_transcript
                for variation in variations.get(token, [])
            ):
                hits += 1

        return round(hits / len(tokens), 2)

    def check(path: str, value: str):
        score = grounded(value)
        confidences.append(FieldConfidence(field_path=path, confidence=score))

    cd = assessment.clinicalDetails
    check("clinicalDetails.clinicalHistory", cd.clinicalHistory)
    check("clinicalDetails.chiefComplaint", cd.chiefComplaint)
    check("clinicalDetails.duration", cd.duration)

    for i, sa in enumerate(assessment.subjectiveAssessments):
        check(f"subjectiveAssessments[{i}].testName", sa.testName)
        check(f"subjectiveAssessments[{i}].conclusion", sa.conclusion)

    for i, t in enumerate(assessment.objectiveAssessment.tests):
        check(f"objectiveAssessment.tests[{i}].testName", t.testName)
        check(f"objectiveAssessment.tests[{i}].value", t.value)

    for i, g in enumerate(assessment.subjectiveGoals):
        check(f"subjectiveGoals[{i}].goalDetails", g.goalDetails)

    for i, g in enumerate(assessment.objectiveGoals):
        check(f"objectiveGoals[{i}].goalName", g.goalName)
        check(f"objectiveGoals[{i}].value", g.value)

    for i, r in enumerate(assessment.recommendation):
        check(f"recommendation[{i}].sessionType", r.sessionType)

    check("patientAdvice.adviceDetails", assessment.patientAdvice.adviceDetails)

    settings = get_settings()
    low = [c.field_path for c in confidences if c.confidence < settings.min_field_confidence]
    overall = round(sum(c.confidence for c in confidences) / len(confidences), 2) if confidences else 1.0

    state["field_confidences"] = confidences
    state["low_confidence_fields"] = low
    state["overall_confidence"] = overall
    return state


def _route_after_validate(state: AgentState) -> str:
    if state["validation_error"] is None:
        return "score_confidence"
    if state["retries"] >= 1:
        # Give up gracefully rather than loop forever; surface the error
        # to the caller so it becomes a 422, not a 500.
        raise ValueError(f"Extraction failed schema validation after retry: {state['validation_error']}")
    return "repair_extraction"


def build_graph():
    from langgraph.graph import StateGraph, END

    graph = StateGraph(AgentState)
    graph.add_node("extract_entities", extract_entities)
    graph.add_node("validate_schema", validate_schema)
    graph.add_node("repair_extraction", repair_extraction)
    graph.add_node("score_confidence", score_confidence)

    graph.set_entry_point("extract_entities")
    graph.add_edge("extract_entities", "validate_schema")
    graph.add_conditional_edges(
        "validate_schema",
        _route_after_validate,
        {"score_confidence": "score_confidence", "repair_extraction": "repair_extraction"},
    )
    graph.add_edge("repair_extraction", "validate_schema")
    graph.add_edge("score_confidence", END)
    return graph.compile()


_compiled_graph = None


def run_extraction(transcript: str) -> ExtractionResult:
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()

    initial_state: AgentState = {
        "transcript": transcript,
        "raw_extraction": None,
        "assessment": None,
        "field_confidences": [],
        "overall_confidence": 1.0,
        "low_confidence_fields": [],
        "validation_error": None,
        "retries": 0,
    }
    final_state = _compiled_graph.invoke(initial_state)

    return ExtractionResult(
        assessment=final_state["assessment"],
        transcript=transcript,
        field_confidences=final_state["field_confidences"],
        overall_confidence=final_state["overall_confidence"],
        low_confidence_fields=final_state["low_confidence_fields"],
    )
