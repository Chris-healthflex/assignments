from typing import Any, TypedDict

from pydantic import Field

from app.models.first_assessment import FirstAssessment, StrictModel


class ExtractionState(TypedDict, total=False):
    transcript: str
    assessment: FirstAssessment
    confidence: dict[str, float]


class ExtractionOutput(StrictModel):
    """Wrapper the model must fill in a single structured-output call.

    Requesting confidence alongside the assessment (rather than as a
    separate pass) keeps this a single deterministic LLM call per
    extraction, matching the graph's one-node design.
    """

    assessment: FirstAssessment
    field_confidence: dict[str, float] = Field(
        description=(
            "Confidence from 0.0 to 1.0 for each top-level assessment field "
            "you populated, keyed by dotted path, e.g. "
            "'clinicalDetails.chiefComplaint', 'objectiveAssessment.tests', "
            "'recommendation'. Score 1.0 only when the transcript states the "
            "value explicitly and unambiguously. Score below 0.5 for anything "
            "paraphrased, inferred, or uncertain. If a field was left empty "
            "because the transcript never mentions it, do not include it here "
            "at all - only score fields you actually populated."
        )
    )


SYSTEM_PROMPT = """You extract clinical assessment data from a clinician-patient transcript.

Return only data explicitly supported by the transcript. Never infer or hallucinate clinical
values, scores, measurements, diagnoses, dates, or recommendations.

Use empty strings for unavailable string fields and empty arrays for unavailable repeated
sections. Preserve all repeated items as arrays - if the transcript mentions multiple tests,
goals, or recommendations, include every one as a separate array entry.

Record a goal, test, or recommendation as soon as the transcript states its name or content,
even if some of its fields (such as a numeric target or a target date) are not mentioned.
Leave those specific unmentioned fields as empty strings rather than omitting the entire
item. For example, a stated treatment goal with no explicit date is still a goal - do not
drop it just because targetDate is missing.

Physiotherapy sessions often state treatment focus areas without using the word "goal" - for
example "emphasis on restoring extension" or "improving ankle mobility." Each such stated focus
area is one objectiveGoals entry: goalName is the focus area itself (e.g. "Restore knee
extension", "Improve ankle mobility"), goalCategory is a short label like "Range of Motion" or
"Strength" if the transcript's wording implies one, otherwise leave it empty. Leave unitName,
value, and targetDate as empty strings when no number or date is stated - populate them only
if the transcript gives one.

Along with the assessment, report your own confidence for each field you populated, per the
field_confidence instructions. Under-report confidence rather than over-report it: if you are
not certain the transcript directly supports a value, score it low so it can be flagged for
human review instead of silently accepted.

The output must conform exactly to the supplied schema."""


def _extract_with_model(state: ExtractionState, model: Any) -> ExtractionState:
    structured = model.with_structured_output(ExtractionOutput)
    result: ExtractionOutput = structured.invoke(
        [
            ("system", SYSTEM_PROMPT),
            ("human", f"Transcript:\n{state['transcript']}"),
        ]
    )
    return {"assessment": result.assessment, "confidence": result.field_confidence}


class ClinicalExtractionGraph:
    def __init__(self, model_name: str, api_key: str | None) -> None:
        self.model_name = model_name
        self.api_key = api_key
        self._graph = None

    def _build(self) -> Any:
        try:
            from langchain_groq import ChatGroq
            from langgraph.graph import END, START, StateGraph
        except ImportError as exc:
            raise RuntimeError("LangGraph extraction dependencies are not installed") from exc

        model = ChatGroq(model=self.model_name, api_key=self.api_key, temperature=0)
        graph = StateGraph(ExtractionState)
        graph.add_node("extract", lambda state: _extract_with_model(state, model))
        graph.add_edge(START, "extract")
        graph.add_edge("extract", END)
        return graph.compile()

    def extract(self, transcript: str) -> ExtractionState:
        if not transcript.strip():
            raise ValueError("Transcript cannot be empty")
        if self._graph is None:
            self._graph = self._build()
        return self._graph.invoke({"transcript": transcript})