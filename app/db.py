from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Optional
from bson import ObjectId
from bson.errors import InvalidId

from app.config import settings
from app.schemas import AssessmentRecord, FirstAssessment

logger = logging.getLogger(__name__)

_mongo_client = None
_db = None
_in_memory_docs: dict[str, dict[str, Any]] = {}


def get_db():
    global _mongo_client, _db
    if _db is None:
        try:
            from motor.motor_asyncio import AsyncIOMotorClient
            _mongo_client = AsyncIOMotorClient(
                settings.MONGODB_URI,
                serverSelectionTimeoutMS=2000
            )
            _db = _mongo_client[settings.MONGODB_DATABASE]
        except Exception as e:
            logger.warning(f"Could not connect to MongoDB at {settings.MONGODB_URI}: {e}. Fallback to mock.")
            _db = None
    return _db


async def save_assessment(assessment: FirstAssessment, meta: Optional[dict[str, Any]] = None) -> AssessmentRecord:
    db = get_db()
    now = datetime.now(timezone.utc)
    doc_data = {
        "assessment": assessment.model_dump(),
        "meta": meta or {},
        "createdAt": now,
    }

    if db is not None:
        try:
            res = await db.assessments.insert_one(doc_data)
            doc_id = str(res.inserted_id)
            return AssessmentRecord(
                id=doc_id,
                createdAt=now,
                assessment=assessment,
                meta=meta
            )
        except Exception as e:
            logger.warning(f"MongoDB save failed, using local store: {e}")

    # In-memory fallback if MongoDB is not reachable
    doc_id = str(ObjectId())
    _in_memory_docs[doc_id] = {
        "_id": doc_id,
        "assessment": assessment.model_dump(),
        "meta": meta or {},
        "createdAt": now,
    }
    return AssessmentRecord(
        id=doc_id,
        createdAt=now,
        assessment=assessment,
        meta=meta
    )


async def get_assessment_by_id(assessment_id: str) -> Optional[AssessmentRecord]:
    try:
        oid = ObjectId(assessment_id)
    except (InvalidId, TypeError):
        return None

    db = get_db()
    if db is not None:
        try:
            doc = await db.assessments.find_one({"_id": oid})
            if doc:
                return AssessmentRecord(
                    id=str(doc["_id"]),
                    createdAt=doc.get("createdAt", datetime.now(timezone.utc)),
                    assessment=FirstAssessment.model_validate(doc["assessment"]),
                    meta=doc.get("meta")
                )
        except Exception as e:
            logger.warning(f"MongoDB fetch failed, checking local store: {e}")

    # Check in-memory store
    if assessment_id in _in_memory_docs:
        doc = _in_memory_docs[assessment_id]
        return AssessmentRecord(
            id=assessment_id,
            createdAt=doc["createdAt"],
            assessment=FirstAssessment.model_validate(doc["assessment"]),
            meta=doc.get("meta")
        )

    return None


async def list_assessments(
    from_date: Optional[datetime] = None,
    to_date: Optional[datetime] = None,
    limit: int = 50,
    skip: int = 0
) -> list[AssessmentRecord]:
    db = get_db()
    query: dict[str, Any] = {}
    if from_date or to_date:
        query["createdAt"] = {}
        if from_date:
            query["createdAt"]["$gte"] = from_date
        if to_date:
            query["createdAt"]["$lte"] = to_date

    results: list[AssessmentRecord] = []

    if db is not None:
        try:
            cursor = db.assessments.find(query).sort("createdAt", -1).skip(skip).limit(limit)
            async for doc in cursor:
                results.append(AssessmentRecord(
                    id=str(doc["_id"]),
                    createdAt=doc.get("createdAt", datetime.now(timezone.utc)),
                    assessment=FirstAssessment.model_validate(doc["assessment"]),
                    meta=doc.get("meta")
                ))
            return results
        except Exception as e:
            logger.warning(f"MongoDB list query failed: {e}")

    # In-memory query fallback
    filtered = []
    for doc in _in_memory_docs.values():
        created = doc["createdAt"]
        if from_date and created < from_date:
            continue
        if to_date and created > to_date:
            continue
        filtered.append(doc)

    filtered.sort(key=lambda x: x["createdAt"], reverse=True)
    sliced = filtered[skip : skip + limit]
    for doc in sliced:
        results.append(AssessmentRecord(
            id=doc["_id"],
            createdAt=doc["createdAt"],
            assessment=FirstAssessment.model_validate(doc["assessment"]),
            meta=doc.get("meta")
        ))
    return results
