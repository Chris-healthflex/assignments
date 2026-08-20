from typing import Any
from app.core.logging import logger
from app.core.errors import ExtractionError
from app.schemas.first_assessment import FirstAssessment
from app.agents.clinical_extraction_graph import create_clinical_extraction_graph, ExtractionState


class ClinicalExtractionService:
    """Service layer running the LangGraph clinical extraction workflow."""

    def __init__(self) -> None:
        self.graph = create_clinical_extraction_graph()

    async def extract_assessment(self, transcript: str) -> FirstAssessment:
        """
        Executes the LangGraph extraction pipeline on a plain text clinical transcript.
        Returns a strictly validated FirstAssessment model or raises ExtractionError.
        """
        logger.info("Starting LangGraph clinical extraction service")
        initial_state: ExtractionState = {
            "transcript": transcript,
            "extracted_data": {},
            "validation_errors": [],
            "confidence_score": 1.0,
            "first_assessment": None,
            "is_valid": True,
        }

        try:
            # LangGraph execution
            final_state = self.graph.invoke(initial_state)

            if not final_state.get("is_valid", False) or final_state.get("first_assessment") is None:
                errors = final_state.get("validation_errors", ["Clinical entity extraction could not be completed."])
                logger.warning("Clinical extraction rejected by graph validation: %s", errors)
                raise ExtractionError(
                    message="Clinical extraction failed validation or fell below confidence threshold.",
                    details={"validation_errors": errors, "confidence": final_state.get("confidence_score", 0.0)}
                )

            assessment: FirstAssessment = final_state["first_assessment"]
            logger.info("Clinical extraction successfully built FirstAssessment schema.")
            return assessment

        except ExtractionError:
            raise
        except Exception as exc:
            logger.exception("Unexpected error during clinical extraction: %s", exc)
            raise ExtractionError(
                message=f"Clinical extraction service failed: {str(exc)}",
                details={"error": str(exc)}
            )


def get_extraction_service() -> ClinicalExtractionService:
    """Dependency provider for ClinicalExtractionService."""
    return ClinicalExtractionService()
