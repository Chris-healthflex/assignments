"""LangGraph clinical entity extraction workflow.

Extracts structured clinical information from clinical session transcripts
using a deterministic, anti-hallucination LangGraph workflow.
"""

from typing import Any, Dict, List, Optional, Tuple, TypedDict
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from app.config import settings
from app.schemas.assessment import FirstAssessment
from app.services.confidence import GroundingCheckResult, validate_grounding
from app.services.prompts import (
    CLINICAL_EXTRACTION_SYSTEM_PROMPT,
    CLINICAL_EXTRACTION_USER_PROMPT,
)


class ExtractionState(TypedDict, total=False):
    """Internal LangGraph state holding transcript and extraction metadata.

    Note: Internal fields (evidence, uncertain_fields, validation_errors) remain
    within this state and are NEVER leaked into the final FirstAssessment schema.
    """

    transcript: str
    raw_assessment: Optional[FirstAssessment]
    evidence: Dict[str, str]
    uncertain_fields: List[Dict[str, Any]]
    validation_errors: List[str]
    final_assessment: Optional[FirstAssessment]
    is_valid: bool


class ClinicalExtractionAgent:
    """Agent orchestrating the LangGraph clinical extraction workflow."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        llm: Optional[Any] = None,
    ) -> None:
        """Initialize ClinicalExtractionAgent.

        Args:
            api_key: Optional OpenAI API key override.
            model: Optional extraction model name override.
            llm: Optional pre-configured LangChain LLM instance (useful for testing).
        """
        self._api_key = api_key or settings.OPENAI_API_KEY
        self.model = model or settings.EXTRACTION_MODEL
        self._llm = llm
        self.graph = self._build_graph()

    @property
    def llm(self) -> Any:
        """Get or lazily initialize structured OpenAI LLM."""
        if self._llm is not None:
            return self._llm

        if not self._api_key or self._api_key.strip() in {
            "",
            "your_openai_api_key_here",
            "mock_key",
        }:
            raise ValueError(
                "OpenAI API key is not configured. Please set OPENAI_API_KEY in environment or .env."
            )

        base_llm = ChatOpenAI(
            model=self.model,
            temperature=0,
            api_key=self._api_key,
        )
        self._llm = base_llm.with_structured_output(FirstAssessment)
        return self._llm

    def _extract_clinical_data_node(self, state: ExtractionState) -> Dict[str, Any]:
        """Node 1: Extract clinical entities from transcript using structured LLM."""
        transcript = state.get("transcript", "").strip()
        if not transcript:
            return {
                "raw_assessment": FirstAssessment(),
                "validation_errors": ["Transcript is empty."],
                "is_valid": False,
            }

        messages = [
            SystemMessage(content=CLINICAL_EXTRACTION_SYSTEM_PROMPT),
            HumanMessage(content=CLINICAL_EXTRACTION_USER_PROMPT.format(transcript=transcript)),
        ]

        try:
            extracted: FirstAssessment = self.llm.invoke(messages)
            if not isinstance(extracted, FirstAssessment):
                # If returned as dict, validate into FirstAssessment
                extracted = FirstAssessment.model_validate(extracted)

            return {
                "raw_assessment": extracted,
                "validation_errors": [],
            }
        except Exception as exc:
            return {
                "raw_assessment": FirstAssessment(),
                "validation_errors": [f"LLM extraction failure: {str(exc)}"],
                "is_valid": False,
            }

    def _validate_grounding_node(self, state: ExtractionState) -> Dict[str, Any]:
        """Node 2: Verify that all extracted clinical values are grounded in the transcript."""
        transcript = state.get("transcript", "")
        raw_assessment = state.get("raw_assessment") or FirstAssessment()
        existing_errors = list(state.get("validation_errors", []))

        grounding_result: GroundingCheckResult = validate_grounding(transcript, raw_assessment)

        all_errors = existing_errors + grounding_result.validation_errors
        is_valid = grounding_result.is_grounded and len(all_errors) == 0

        return {
            "evidence": grounding_result.evidence,
            "uncertain_fields": grounding_result.uncertain_fields,
            "validation_errors": all_errors,
            "is_valid": is_valid,
        }

    def _normalize_extraction_node(self, state: ExtractionState) -> Dict[str, Any]:
        """Node 3: Normalize extraction into production FirstAssessment schema."""
        raw_assessment = state.get("raw_assessment") or FirstAssessment()

        # Ensure final assessment is a clean FirstAssessment instance
        # Dump and re-validate to guarantee strict adherence to schema defaults and forbid extras
        final_assessment = FirstAssessment.model_validate(raw_assessment.model_dump())

        return {
            "final_assessment": final_assessment,
        }

    def _build_graph(self) -> Any:
        """Construct the LangGraph workflow."""
        workflow = StateGraph(ExtractionState)

        # Add Nodes
        workflow.add_node("extract_clinical_data", self._extract_clinical_data_node)
        workflow.add_node("validate_grounding", self._validate_grounding_node)
        workflow.add_node("normalize_extraction", self._normalize_extraction_node)

        # Add Edges
        workflow.add_edge(START, "extract_clinical_data")
        workflow.add_edge("extract_clinical_data", "validate_grounding")
        workflow.add_edge("validate_grounding", "normalize_extraction")
        workflow.add_edge("normalize_extraction", END)

        return workflow.compile()

    def extract(self, transcript: str) -> ExtractionState:
        """Execute the LangGraph workflow for a given transcript.

        Args:
            transcript: Source clinical transcript text.

        Returns:
            Final ExtractionState containing final_assessment and validation metadata.
        """
        initial_state: ExtractionState = {
            "transcript": transcript,
            "raw_assessment": None,
            "evidence": {},
            "uncertain_fields": [],
            "validation_errors": [],
            "final_assessment": None,
            "is_valid": True,
        }
        final_state = self.graph.invoke(initial_state)
        return final_state


def run_clinical_extraction(
    transcript: str,
    agent: Optional[ClinicalExtractionAgent] = None,
) -> Tuple[FirstAssessment, ExtractionState]:
    """Convenience function to run clinical extraction workflow.

    Args:
        transcript: Clinical session transcript text.
        agent: Optional ClinicalExtractionAgent instance.

    Returns:
        Tuple of (FirstAssessment production schema, complete internal ExtractionState).
    """
    runner = agent or ClinicalExtractionAgent()
    state = runner.extract(transcript)
    assessment = state.get("final_assessment") or FirstAssessment()
    return assessment, state
