from __future__ import annotations

from datetime import datetime
from typing import Any

from app.agents.assessment_graph import AssessmentGraphResult, run_assessment_graph
from app.db.repository import AssessmentRepository
from app.schemas.assessment import FirstAssessment
from app.services.transcription import transcribe_wav


class ConfidenceTooLowError(Exception):
    """Raised when one or more fields fail the confidence gate."""

    def __init__(self, low_confidence_fields: list[dict[str, Any]]):
        self.low_confidence_fields = low_confidence_fields
        super().__init__("Extraction confidence below threshold for one or more fields.")


def parse_wav_to_assessment(file_bytes: bytes, filename: str) -> FirstAssessment:
    """Full AI pipeline: WAV bytes -> transcript -> FirstAssessment.

    Raises ConfidenceTooLowError if any field fails the confidence gate
    (caller should translate this into an HTTP 422).
    """
    transcript = transcribe_wav(file_bytes, filename)
    result: AssessmentGraphResult = run_assessment_graph(transcript)

    if not result.passed:
        raise ConfidenceTooLowError(
            [f.model_dump() for f in result.low_confidence_fields]
        )

    assert result.assessment is not None
    return result.assessment


async def save_assessment(assessment: FirstAssessment) -> str:
    return await AssessmentRepository.save(assessment)


async def get_assessment(assessment_id: str) -> dict[str, Any] | None:
    return await AssessmentRepository.get_by_id(assessment_id)


async def list_assessments(
    date_from: datetime | None = None,
    date_to: datetime | None = None,
) -> list[dict[str, Any]]:
    return await AssessmentRepository.list_all(date_from=date_from, date_to=date_to)
