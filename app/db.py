"""
MongoDB persistence (async, via motor).

Collection: `assessments`
Document shape:
    {
      _id:        ObjectId,
      createdAt:  datetime (UTC),
      assessment: <FirstAssessment JSON, exact schema>,
      meta: { sourceFile, transcript, flags[], overallConfidence }   # optional audit trail
    }
The API always returns `assessment` untouched so the frontend payload
stays an exact schema match; `id`/`createdAt` are returned alongside it.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from bson import ObjectId
from bson.errors import InvalidId
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorCollection
from pydantic import BaseModel, Field

from .config import settings
from .schemas import FieldFlag, FirstAssessment

_client: AsyncIOMotorClient | None = None


def get_client() -> AsyncIOMotorClient:
    global _client
    if _client is None:
        _client = AsyncIOMotorClient(settings.mongodb_uri, serverSelectionTimeoutMS=5000)
    return _client


def collection() -> AsyncIOMotorCollection:
    return get_client()[settings.mongodb_db]["assessments"]


async def ensure_indexes() -> None:
    await collection().create_index("createdAt")


async def close() -> None:
    global _client
    if _client is not None:
        _client.close()
        _client = None


# --------------------------------------------------------------------------- models
class AssessmentMeta(BaseModel):
    sourceFile: str = ""
    transcript: str = ""
    flags: list[FieldFlag] = Field(default_factory=list)
    overallConfidence: float | None = None


class SaveAssessmentRequest(BaseModel):
    """Body for POST /assessments."""

    assessment: FirstAssessment
    meta: AssessmentMeta | None = None


class StoredAssessment(BaseModel):
    """Response for GET endpoints and POST /assessments."""

    id: str
    createdAt: datetime
    assessment: FirstAssessment
    meta: AssessmentMeta | None = None


def _to_model(doc: dict[str, Any]) -> StoredAssessment:
    return StoredAssessment(
        id=str(doc["_id"]),
        createdAt=doc["createdAt"],
        assessment=FirstAssessment.model_validate(doc["assessment"]),
        meta=AssessmentMeta.model_validate(doc["meta"]) if doc.get("meta") else None,
    )


# --------------------------------------------------------------------------- ops
async def save_assessment(assessment: FirstAssessment, meta: AssessmentMeta | None = None) -> StoredAssessment:
    doc = {
        "createdAt": datetime.now(timezone.utc),
        "assessment": assessment.model_dump(),
        "meta": meta.model_dump() if meta else None,
    }
    res = await collection().insert_one(doc)
    doc["_id"] = res.inserted_id
    return _to_model(doc)


async def get_assessment(assessment_id: str) -> StoredAssessment | None:
    try:
        oid = ObjectId(assessment_id)
    except InvalidId:
        return None
    doc = await collection().find_one({"_id": oid})
    return _to_model(doc) if doc else None


async def list_assessments(
    from_date: datetime | None = None,
    to_date: datetime | None = None,
    limit: int = 100,
    skip: int = 0,
) -> list[StoredAssessment]:
    query: dict[str, Any] = {}
    if from_date or to_date:
        query["createdAt"] = {}
        if from_date:
            query["createdAt"]["$gte"] = from_date
        if to_date:
            query["createdAt"]["$lte"] = to_date
    cursor = collection().find(query).sort("createdAt", -1).skip(skip).limit(limit)
    return [_to_model(d) async for d in cursor]
