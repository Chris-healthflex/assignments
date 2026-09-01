import json
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from typing import TypedDict


class ExtractionState(TypedDict):
    transcript: str
    raw_extraction: dict
    error: str


EXTRACTION_PROMPT = """You are a clinical data extraction assistant.
You will be given a transcript of a clinician-patient session.

Extract ONLY information that is explicitly stated in the transcript.
Do NOT infer, guess, or fabricate any clinical values, scores, dates, or facts.
If a piece of information is not clearly present in the transcript, leave it as an empty string "" (for text) or an empty list [] (for lists).

Return a single JSON object with exactly this structure (no extra keys, no renamed keys):

{{
  "clinicalHistory": "",
  "chiefComplaint": "",
  "duration": "",
  "subjectiveAssessments": [{{"testName": "", "conclusion": ""}}],
  "objectiveTests": [{{"testName": "", "unitName": "", "value": "", "left": "", "right": "", "comments": ""}}],
  "subjectiveGoals": [{{"goalDetails": "", "targetDate": ""}}],
  "objectiveGoals": [{{"goalName": "", "goalCategory": "", "unitName": "", "value": "", "targetDate": ""}}],
  "recommendation": [{{"sessionType": "", "sessionFrequency": ""}}],
  "adviceDetails": ""
}}

Transcript:
{transcript}

Return ONLY the JSON object, no markdown, no commentary.
"""


def extract_node(state: ExtractionState) -> ExtractionState:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = EXTRACTION_PROMPT.format(transcript=state["transcript"])

    response = llm.invoke(prompt)
    content = response.content.strip()

    # Strip accidental markdown fences
    if content.startswith("```"):
        content = content.strip("`")
        if content.startswith("json"):
            content = content[4:].strip()

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as e:
        return {**state, "error": f"Failed to parse LLM output as JSON: {e}"}

    return {**state, "raw_extraction": parsed, "error": ""}


def build_extraction_graph():
    graph = StateGraph(ExtractionState)
    graph.add_node("extract", extract_node)
    graph.set_entry_point("extract")
    graph.add_edge("extract", END)
    return graph.compile()


def extract_clinical_data(transcript: str) -> dict:
    """
    Runs the LangGraph extraction agent on a transcript.
    Returns the raw extracted dict (pre-mapping to FirstAssessment schema).
    Raises ValueError if extraction failed or produced invalid JSON.
    """
    app = build_extraction_graph()
    result = app.invoke({"transcript": transcript, "raw_extraction": {}, "error": ""})

    if result["error"]:
        raise ValueError(result["error"])

    return result["raw_extraction"]