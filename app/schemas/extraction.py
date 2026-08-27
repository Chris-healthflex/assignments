"""What the extraction agent returns: the assessment plus its confidence verdict.

This is the model the LLM is asked to produce, so the confidence fields are part
of the structured output rather than something computed afterwards. It is
internal to the pipeline - the API returns only `FirstAssessment`.

The confidence defaults fail closed. They apply only when the model omits the
field, which it does on an empty transcript - and a defaulted-confident result
would be returned as HTTP 200 with an empty assessment. Defaulting to 0.0 keeps
the extracted payload available as `partial_assessment` in the 422 body rather
than discarding it, which a hard validation error would do.
"""

from typing import List

from pydantic import BaseModel, Field

from app.schemas.assessment import FirstAssessment


class ExtractionResult(BaseModel):
    assessment: FirstAssessment
    confidence_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Overall confidence score of extracted clinical data (0.0 to 1.0)."
    )
    is_confident: bool = Field(
        default=False,
        description="Whether overall extraction confidence exceeds minimum threshold."
    )
    field_errors: List[str] = Field(
        default_factory=list,
        description="List of unconfident, missing, or potentially ambiguous clinical field details."
    )
