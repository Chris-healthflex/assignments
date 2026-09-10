"""Exception hierarchy for the pipeline.

Everything descends from ClinicalAssessmentError, so one clause catches the lot
while the API layer still maps each type to its own status code.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Type-only, so the errors have no runtime dependency on grounding.
    from collections.abc import Sequence

    from .grounding import FieldVerdict


class ClinicalAssessmentError(Exception):
    """Base class for every error raised by this package."""


class AudioDecodeError(ClinicalAssessmentError):
    """Raised when an upload is not decodable WAV audio."""


class TranscriptionError(ClinicalAssessmentError):
    """Raised when speech-to-text fails to produce a transcript."""


class ExtractionError(ClinicalAssessmentError):
    """Raised when the extraction model fails or refuses to answer."""


class LowConfidenceError(ClinicalAssessmentError):
    """Raised when too few extracted fields are grounded in the transcript.

    Carries what the API needs for a 422 that names the failed fields.
    """

    def __init__(
        self,
        confidence: float,
        threshold: float,
        ungrounded_fields: "Sequence[FieldVerdict]",
    ) -> None:
        super().__init__(
            f"Extraction confidence {confidence:.2f} is below the required "
            f"{threshold:.2f}; {len(ungrounded_fields)} field(s) could not be "
            "verified against the transcript"
        )
        self.confidence = confidence
        self.threshold = threshold
        self.ungrounded_fields = tuple(ungrounded_fields)


class StorageError(ClinicalAssessmentError):
    """Raised when the assessment store is unreachable or rejects a write."""


class AssessmentNotFoundError(ClinicalAssessmentError):
    """Raised when no stored assessment matches the requested identifier."""
