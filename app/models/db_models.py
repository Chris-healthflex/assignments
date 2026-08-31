"""Mongo-facing document model. Kept separate from FirstAssessment so the
storage envelope (id, timestamps, source metadata) never leaks into the
strict frontend schema."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field

from app.models.schema import FirstAssessment


class AssessmentDocument(BaseModel):
    id: Optional[str] = Field(default=None, alias="_id")
    assessment: FirstAssessment
    transcript: str
    audio_filename: Optional[str] = None
    overall_confidence: float = 1.0
    low_confidence_fields: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    model_config = {"populate_by_name": True}

    def to_mongo(self) -> dict:
        doc = self.model_dump(by_alias=True, exclude={"id"})
        return doc
