from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field

from app.schemas.first_assessment import FirstAssessment
from app.db.mongo import get_assessments_collection


class AssessmentRecord(BaseModel):
    """What's actually stored in Mongo: the schema-exact assessment plus
    server-side metadata. The metadata never leaks into GET responses' `assessment`
    key, so the frontend still gets an exact FirstAssessment back."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    assessment: FirstAssessment
    overall_confidence: float = 1.0
    extraction_flags: list[str] = Field(default_factory=list)
    source_audio_filename: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


async def save_assessment(record: AssessmentRecord) -> str:
    coll = get_assessments_collection()
    doc = record.model_dump()
    doc["_id"] = doc.pop("id")
    await coll.insert_one(doc)
    return doc["_id"]


async def get_assessment(assessment_id: str) -> Optional[dict]:
    coll = get_assessments_collection()
    doc = await coll.find_one({"_id": assessment_id})
    if doc is None:
        return None
    doc["id"] = doc.pop("_id")
    return doc


async def list_assessments(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    limit: int = 50,
    skip: int = 0,
) -> list[dict]:
    coll = get_assessments_collection()
    query: dict = {}
    if start_date or end_date:
        date_filter = {}
        if start_date:
            date_filter["$gte"] = start_date
        if end_date:
            date_filter["$lte"] = end_date
        query["created_at"] = date_filter

    cursor = coll.find(query).sort("created_at", -1).skip(skip).limit(limit)
    results = []
    async for doc in cursor:
        doc["id"] = doc.pop("_id")
        results.append(doc)
    return results
