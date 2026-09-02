"""Database persistence document models.

Note: AssessmentDocument is a persistence-layer wrapper containing metadata (_id, created_at).
The FirstAssessment schema itself remains untouched and pure without any database fields.
"""

from datetime import datetime, timezone
from typing import Optional
from pydantic import BaseModel, ConfigDict, Field

from app.schemas.assessment import FirstAssessment


class AssessmentDocument(BaseModel):
    """Database persistence model for storing clinical assessments."""

    model_config = ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed=True,
    )

    id: str = Field(description="MongoDB ObjectId string representation")
    assessment: FirstAssessment = Field(description="Pure FirstAssessment clinical payload")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    source_audio: Optional[str] = Field(default=None, description="Optional source audio filename")
