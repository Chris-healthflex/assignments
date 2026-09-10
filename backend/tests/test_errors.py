"""Tests for the package exception hierarchy."""

import pytest

from clinical_assessment.errors import (
    AssessmentNotFoundError,
    AudioDecodeError,
    ClinicalAssessmentError,
    ExtractionError,
    LowConfidenceError,
    StorageError,
    TranscriptionError,
)
from clinical_assessment.grounding import FieldVerdict, GroundingFailure

# LowConfidenceError is excluded here because it takes a structured payload
# rather than a message; it gets its own tests below.
MESSAGE_ONLY_ERRORS = [
    AudioDecodeError,
    TranscriptionError,
    ExtractionError,
    StorageError,
    AssessmentNotFoundError,
]

UNVERIFIED_FIELD = FieldVerdict(
    field_path="objectiveAssessment.tests[0].value",
    value="45",
    evidence="flexion was measured",
    failure=GroundingFailure.DIGITS_NOT_IN_EVIDENCE,
)


@pytest.mark.parametrize(
    "error_type", MESSAGE_ONLY_ERRORS, ids=lambda cls: cls.__name__
)
def test_every_error_is_catchable_as_clinical_assessment_error(
    error_type: type[ClinicalAssessmentError],
) -> None:
    with pytest.raises(ClinicalAssessmentError):
        raise error_type("something went wrong")


def test_low_confidence_error_is_catchable_as_clinical_assessment_error() -> None:
    with pytest.raises(ClinicalAssessmentError):
        raise LowConfidenceError(0.5, 0.7, [UNVERIFIED_FIELD])


def test_low_confidence_error_carries_the_fields_that_failed() -> None:
    error = LowConfidenceError(0.5, 0.7, [UNVERIFIED_FIELD])

    assert error.ungrounded_fields == (UNVERIFIED_FIELD,)


def test_low_confidence_message_names_the_two_ratios() -> None:
    error = LowConfidenceError(0.5, 0.7, [UNVERIFIED_FIELD])

    assert "0.50" in str(error) and "0.70" in str(error)


def test_base_error_is_not_raised_for_an_unrelated_exception() -> None:
    with pytest.raises(ValueError):
        raise ValueError("unrelated")
