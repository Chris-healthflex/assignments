from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from app.config import get_settings
from app.services.extraction import FieldConfidence


class ConfidenceErrorDetail(BaseModel):
    model_config = ConfigDict(extra="forbid")

    field: str
    confidence: float = Field(
        ge=0.0,
        le=1.0,
    )
    threshold: float
    reason: str


class ExtractionConfidenceError(Exception):
    def __init__(
        self,
        details: list[ConfidenceErrorDetail],
    ):
        self.details = details

        super().__init__(
            "One or more extracted clinical fields "
            "did not meet the required confidence threshold."
        )


def _field_has_value(
    assessment,
    field_path: str,
) -> bool:
    """
    Determine whether a production field contains extracted data.

    Missing/empty fields are allowed and should not automatically
    cause a 422 response.
    """

    try:
        value = assessment

        parts = field_path.split(".")

        for part in parts:
            if "[" in part:
                name = part.split("[", 1)[0]
                index_text = part.split("[", 1)[1].split("]", 1)[0]

                value = getattr(
                    value,
                    name,
                )

                value = value[
                    int(index_text)
                ]

            else:
                value = getattr(
                    value,
                    part,
                )

        if isinstance(
            value,
            str,
        ):
            return bool(value.strip())

        if isinstance(
            value,
            list,
        ):
            return len(value) > 0

        return value is not None

    except (
        AttributeError,
        IndexError,
        KeyError,
        ValueError,
        TypeError,
    ):
        return False


def validate_confidence(
    confidence: list[FieldConfidence],
    assessment=None,
) -> None:
    settings = get_settings()

    failures: list[ConfidenceErrorDetail] = []

    for item in confidence:

        # If the field was not actually populated, there is
        # nothing to validate.
        if assessment is not None:
            if not _field_has_value(
                assessment,
                item.field,
            ):
                continue

        if item.confidence < settings.confidence_threshold:
            failures.append(
                ConfidenceErrorDetail(
                    field=item.field,
                    confidence=item.confidence,
                    threshold=settings.confidence_threshold,
                    reason=item.reason,
                )
            )

    if failures:
        raise ExtractionConfidenceError(
            failures
        )
