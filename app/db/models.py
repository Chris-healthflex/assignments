"""MongoDB document models (D4).

The critical property here is that the stored ``assessment`` sub-document is
byte-identical to what the frontend consumes. Metadata - transcript, timings,
confidence, source filename - lives beside it, never inside it. Flattening
metadata into the assessment would be convenient for querying and would break
the schema contract the frontend depends on, so it is kept out.

MongoDB's ``_id`` is an ObjectId, which is not JSON-serialisable. Conversion to
and from a string happens at this boundary, so neither the API layer nor the
repository has to think about BSON types.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.extraction.confidence import FieldFlag
from app.schemas.first_assessment import FirstAssessment


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class AssessmentMetadata(BaseModel):
    """Everything about a parse that is not part of the assessment itself."""

    model_config = ConfigDict(extra="ignore")

    sourceFilename: str = ""
    transcript: str = ""
    transcriptLanguage: str = ""
    audioDurationSeconds: float = 0.0
    whisperModel: str = ""
    whisperBackend: str = ""
    llmProvider: str = ""
    llmModel: str = ""
    confidence: float = 0.0
    confidenceThreshold: float = 0.0
    flaggedFields: list[FieldFlag] = Field(default_factory=list)
    rejectedCount: int = 0
    sectionScores: dict[str, float] = Field(default_factory=dict)
    timings: dict[str, float] = Field(default_factory=dict)


class StoredAssessment(BaseModel):
    """An assessment as returned from the database."""

    model_config = ConfigDict(extra="ignore")

    id: str
    createdAt: datetime
    assessment: FirstAssessment
    metadata: AssessmentMetadata = Field(default_factory=AssessmentMetadata)


def to_document(
    assessment: FirstAssessment,
    metadata: AssessmentMetadata | None = None,
    *,
    created_at: datetime | None = None,
) -> dict[str, Any]:
    """Build the BSON document to insert.

    ``model_dump`` on the assessment keeps it exactly seven keys wide, so the
    stored copy round-trips through ``FirstAssessment`` unchanged.
    """
    return {
        "createdAt": created_at or utcnow(),
        "assessment": assessment.model_dump(),
        "metadata": (metadata or AssessmentMetadata()).model_dump(),
    }


def from_document(document: dict[str, Any]) -> StoredAssessment:
    """Convert a raw Mongo document into the API-facing model."""
    created = document.get("createdAt") or utcnow()
    if isinstance(created, datetime) and created.tzinfo is None:
        # MongoDB returns naive UTC datetimes; make that explicit so the API
        # does not emit timestamps that look local.
        created = created.replace(tzinfo=timezone.utc)

    return StoredAssessment(
        id=str(document.get("_id", "")),
        createdAt=created,
        assessment=FirstAssessment.model_validate(document.get("assessment") or {}),
        metadata=AssessmentMetadata.model_validate(document.get("metadata") or {}),
    )
