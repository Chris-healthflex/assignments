import json
import logging
from dataclasses import dataclass, field
from typing import Any, List, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from app.config import Settings
from app.errors import PipelineError
from app.models import FirstAssessment
from app.prompts import (
    EXTRACTION_SYSTEM,
    EXTRACTION_USER,
    VERIFICATION_SYSTEM,
    VERIFICATION_USER,
)

logger = logging.getLogger(__name__)

MIN_TRANSCRIPT_CHARS = 40


class DraftClinicalDetails(BaseModel):
    clinicalHistory: Optional[str] = Field(
        None, description="Relevant past medical, surgical or injury history."
    )
    chiefComplaint: Optional[str] = Field(
        None, description="The main problem the patient reports."
    )
    duration: Optional[str] = Field(
        None, description="How long the chief complaint has been present."
    )


class DraftSubjectiveAssessment(BaseModel):
    testName: Optional[str] = Field(
        None, description="Name of the subjective test, scale or questionnaire."
    )
    conclusion: Optional[str] = Field(
        None, description="Reported result or interpretation of that test."
    )


class DraftObjectiveTest(BaseModel):
    testName: Optional[str] = Field(None, description="Name of the measured test.")
    unitName: Optional[str] = Field(
        None, description="Unit of measurement, for example degrees, cm, seconds."
    )
    value: Optional[str] = Field(
        None, description="Measured value when the test has no side."
    )
    left: Optional[str] = Field(None, description="Value measured on the left side.")
    right: Optional[str] = Field(None, description="Value measured on the right side.")
    comments: Optional[str] = Field(
        None, description="Remarks stated about this measurement."
    )


class DraftSubjectiveGoal(BaseModel):
    goalDetails: Optional[str] = Field(None, description="The goal as described.")
    targetDate: Optional[str] = Field(
        None, description="Target date or timeframe exactly as stated."
    )


class DraftObjectiveGoal(BaseModel):
    goalName: Optional[str] = Field(None, description="Name of the measurable goal.")
    goalCategory: Optional[str] = Field(
        None, description="Category such as range of motion or strength."
    )
    unitName: Optional[str] = Field(None, description="Unit for the target value.")
    value: Optional[str] = Field(None, description="Target value to reach.")
    targetDate: Optional[str] = Field(
        None, description="Target date or timeframe exactly as stated."
    )


class DraftRecommendation(BaseModel):
    sessionType: Optional[str] = Field(
        None, description="Type of session or therapy recommended."
    )
    sessionFrequency: Optional[str] = Field(
        None, description="How often the sessions should happen."
    )


class DraftPatientAdvice(BaseModel):
    adviceDetails: Optional[str] = Field(
        None, description="Home advice, precautions or instructions given."
    )


class AssessmentDraft(BaseModel):
    """Extraction target. Every field is optional so the model can leave out
    anything the transcript does not state instead of inventing a value."""

    clinicalDetails: DraftClinicalDetails = DraftClinicalDetails()
    subjectiveAssessments: List[DraftSubjectiveAssessment] = Field(default_factory=list)
    objectiveTests: List[DraftObjectiveTest] = Field(default_factory=list)
    subjectiveGoals: List[DraftSubjectiveGoal] = Field(default_factory=list)
    objectiveGoals: List[DraftObjectiveGoal] = Field(default_factory=list)
    recommendation: List[DraftRecommendation] = Field(default_factory=list)
    patientAdvice: DraftPatientAdvice = DraftPatientAdvice()


class Verification(BaseModel):
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="How well the extracted values are supported by the transcript.",
    )
    unsupportedFields: List[str] = Field(
        default_factory=list,
        description="Dotted paths of values the transcript does not state.",
    )
    notes: str = Field("", description="Short explanation of the score.")


@dataclass
class ExtractionResult:
    assessment: FirstAssessment
    confidence: float
    unextracted_fields: List[str] = field(default_factory=list)
    unsupported_fields: List[str] = field(default_factory=list)
    notes: str = ""


class State(TypedDict, total=False):
    transcript: str
    draft: AssessmentDraft
    verification: Verification


def build_llm(settings: Settings) -> BaseChatModel:
    if not settings.google_api_key:
        raise PipelineError(
            "extraction_failed",
            "GOOGLE_API_KEY is required to run the extraction workflow.",
            502,
            [{"field": "google_api_key", "message": "missing api key"}],
        )
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=settings.llm_model, temperature=0, api_key=settings.google_api_key
    )


def build_graph(llm: BaseChatModel):
    """Two-step workflow: extract the assessment, then check it against the
    transcript to get the confidence the /parse threshold uses."""
    extract_chain = ChatPromptTemplate.from_messages(
        [("system", EXTRACTION_SYSTEM), ("human", EXTRACTION_USER)]
    ) | llm.with_structured_output(AssessmentDraft)
    verify_chain = ChatPromptTemplate.from_messages(
        [("system", VERIFICATION_SYSTEM), ("human", VERIFICATION_USER)]
    ) | llm.with_structured_output(Verification)

    async def extract(state: State) -> State:
        try:
            draft = await extract_chain.ainvoke({"transcript": state["transcript"]})
        except Exception as exc:
            raise PipelineError(
                "extraction_failed",
                "The extraction model call failed.",
                502,
                [{"field": "transcript", "message": str(exc)}],
            ) from exc
        return {"draft": draft}

    async def verify(state: State) -> State:
        extraction = json.dumps(state["draft"].model_dump(exclude_none=True), indent=2)
        try:
            verification = await verify_chain.ainvoke(
                {"transcript": state["transcript"], "extraction": extraction}
            )
        except Exception as exc:
            raise PipelineError(
                "extraction_failed",
                "The extraction verification call failed.",
                502,
                [{"field": "transcript", "message": str(exc)}],
            ) from exc
        return {"verification": verification}

    graph = StateGraph(State)
    graph.add_node("extract", extract)
    graph.add_node("verify", verify)
    graph.add_edge(START, "extract")
    graph.add_edge("extract", "verify")
    graph.add_edge("verify", END)
    return graph.compile()


def _text(value: Optional[str]) -> str:
    return (value or "").strip()


def _entries(items: List[Any], fields: List[str]) -> List[dict]:
    """Map draft list items onto the schema fields, dropping empty entries."""
    mapped = []
    for item in items:
        data = item.model_dump()
        entry = {name: _text(data.get(name)) for name in fields}
        if any(entry.values()):
            mapped.append(entry)
    return mapped


def to_assessment(draft: AssessmentDraft) -> FirstAssessment:
    return FirstAssessment.model_validate(
        {
            "clinicalDetails": {
                "clinicalHistory": _text(draft.clinicalDetails.clinicalHistory),
                "chiefComplaint": _text(draft.clinicalDetails.chiefComplaint),
                "duration": _text(draft.clinicalDetails.duration),
            },
            "subjectiveAssessments": _entries(
                draft.subjectiveAssessments, ["testName", "conclusion"]
            ),
            "objectiveAssessment": {
                "tests": _entries(
                    draft.objectiveTests,
                    ["testName", "unitName", "value", "left", "right", "comments"],
                )
            },
            "subjectiveGoals": _entries(
                draft.subjectiveGoals, ["goalDetails", "targetDate"]
            ),
            "objectiveGoals": _entries(
                draft.objectiveGoals,
                ["goalName", "goalCategory", "unitName", "value", "targetDate"],
            ),
            "recommendation": _entries(
                draft.recommendation, ["sessionType", "sessionFrequency"]
            ),
            "patientAdvice": {"adviceDetails": _text(draft.patientAdvice.adviceDetails)},
        }
    )


def find_unextracted_fields(assessment: FirstAssessment) -> List[str]:
    """Dotted paths of the fields that stayed empty after extraction."""
    unextracted: List[str] = []

    def visit(value: Any, path: str) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                visit(child, f"{path}.{key}" if path else key)
        elif isinstance(value, list):
            if not value:
                unextracted.append(path)
            for index, child in enumerate(value):
                visit(child, f"{path}[{index}]")
        elif value == "":
            unextracted.append(path)

    visit(assessment.model_dump(), "")
    return unextracted


async def extract_assessment(
    transcript: str, settings: Settings, llm: Optional[BaseChatModel] = None
) -> ExtractionResult:
    transcript = transcript.strip()
    if len(transcript) < MIN_TRANSCRIPT_CHARS:
        raise PipelineError(
            "extraction_failed",
            "The transcript is too short to extract a clinical assessment.",
            422,
            [
                {
                    "field": "transcript",
                    "message": f"expected at least {MIN_TRANSCRIPT_CHARS} characters, "
                    f"got {len(transcript)}",
                }
            ],
        )

    graph = build_graph(llm or build_llm(settings))
    state = await graph.ainvoke({"transcript": transcript})
    verification: Verification = state["verification"]
    assessment = to_assessment(state["draft"])
    result = ExtractionResult(
        assessment=assessment,
        confidence=verification.confidence,
        unextracted_fields=find_unextracted_fields(assessment),
        unsupported_fields=verification.unsupportedFields,
        notes=verification.notes,
    )

    threshold = settings.extraction_confidence_threshold
    if result.confidence < threshold:
        raise PipelineError(
            "low_extraction_confidence",
            f"Extraction confidence {result.confidence:.2f} is below the required "
            f"threshold of {threshold:.2f}.",
            422,
            [
                {
                    "field": "confidence",
                    "message": result.notes
                    or "the transcript does not support a reliable extraction",
                    "value": result.confidence,
                    "threshold": threshold,
                }
            ]
            + [
                {"field": path, "message": "not supported by the transcript"}
                for path in result.unsupported_fields
            ],
        )

    logger.info(
        "extraction confidence=%.2f unextracted=%d unsupported=%d",
        result.confidence,
        len(result.unextracted_fields),
        len(result.unsupported_fields),
    )
    return result
