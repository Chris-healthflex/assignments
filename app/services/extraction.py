import json
from langchain_groq import ChatGroq # type: ignore
from langgraph.graph import StateGraph, END
from typing import TypedDict
from app.core.config import GROQ_API_KEY


class ExtractionState(TypedDict):
    transcript: str
    raw_extraction: dict
    error: str


EXTRACTION_PROMPT = """You are a clinical data extraction assistant.
You will be given a transcript of a clinician-patient session. This transcript
was produced by automatic speech-to-text and may contain garbled, misheard,
or nonsensical words or phrases (e.g. "negic 5", "Butella mobility", "ankle
dose of flexion", "condolose fracture"). These are transcription errors, not
real clinical terms.

Rules — follow strictly:
1. Extract ONLY information that is explicitly and clearly stated in the transcript.
2. Do NOT infer, guess, fabricate, or "correct" any clinical values, scores, dates, or facts.
3. If a word or phrase is garbled, unclear, or does not make clinical sense, do NOT
   silently substitute what you think was "probably meant." Leave the specific
   field empty instead.
4. Only convert a garbled phrase into a clean value if the surrounding numeric/
   contextual evidence makes the meaning unambiguous (e.g. a number clearly
   paired with "degrees" next to a named joint movement). If there is real doubt,
   leave it blank rather than guess.
5. If a piece of information is not clearly present in the transcript, leave it as
   an empty string "" (for text) or an empty list [] (for lists). Never pad lists
   with placeholder/empty objects — omit the item entirely instead.

Return a single JSON object with exactly this structure (no extra keys, no renamed keys):

{{
  \"clinicalHistory\": \"\",
  \"chiefComplaint\": \"\",
  \"duration\": \"\",
  \"subjectiveAssessments\": [{{\"testName\": \"\", \"conclusion\": \"\"}}],
  \"objectiveTests\": [{{\"testName\": \"\", \"unitName\": \"\", \"value\": \"\", \"left\": \"\", \"right\": \"\", \"comments\": \"\"}}],
  \"subjectiveGoals\": [{{\"goalDetails\": \"\", \"targetDate\": \"\"}}],
  \"objectiveGoals\": [{{\"goalName\": \"\", \"goalCategory\": \"\", \"unitName\": \"\", \"value\": \"\", \"targetDate\": \"\"}}],
  \"recommendation\": [{{\"sessionType\": \"\", \"sessionFrequency\": \"\"}}],
  \"adviceDetails\": \"\"
}}

Transcript:
{transcript}

Return ONLY the JSON object, no markdown, no commentary.
"""


def extract_node(state: ExtractionState) -> ExtractionState:
    llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0, api_key=GROQ_API_KEY)
    prompt = EXTRACTION_PROMPT.format(transcript=state["transcript"])

    response = llm.invoke(prompt)
    content = response.content.strip()

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
