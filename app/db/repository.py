"""Persistence for assessments. Isolates all Mongo/BSON details from the API."""
from __future__ import annotations

from datetime import datetime, time, timezone
from typing import Any, Dict, List, Optional

from bson import ObjectId
from bson.errors import InvalidId

from app.config import get_settings
from app.db.connection import get_database
from app.models.assessment import FirstAssessment
from app.models.internal import StoredAssessment


class InvalidAssessmentId(Exception):
    """Raised when a supplied id is not a valid Mongo ObjectId."""


def _to_stored(doc: Dict[str, Any]) -> StoredAssessment:
    return StoredAssessment(
        id=str(doc["_id"]),
        assessment=FirstAssessment.model_validate(doc["assessment"]),
        created_at=doc.get("created_at") or datetime.now(timezone.utc),
        source_transcript=doc.get("source_transcript"),
    )


def _parse_object_id(assessment_id: str) -> ObjectId:
    try:
        return ObjectId(assessment_id)
    except (InvalidId, TypeError) as exc:
        raise InvalidAssessmentId(assessment_id) from exc


class AssessmentRepository:
    def __init__(self, collection=None) -> None:
        if collection is not None:
            self._collection = collection
        else:
            settings = get_settings()
            self._collection = get_database()[settings.assessments_collection]

    async def save(
        self, assessment: FirstAssessment, source_transcript: Optional[str] = None
    ) -> StoredAssessment:
        document = {
            "assessment": assessment.model_dump(mode="json"),
            "created_at": datetime.now(timezone.utc),
            "source_transcript": source_transcript,
        }
        result = await self._collection.insert_one(document)
        document["_id"] = result.inserted_id
        return _to_stored(document)

    async def get_by_id(self, assessment_id: str) -> Optional[StoredAssessment]:
        oid = _parse_object_id(assessment_id)
        doc = await self._collection.find_one({"_id": oid})
        return _to_stored(doc) if doc else None

    async def list(
        self,
        date: Optional[str] = None,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        limit: int = 100,
        skip: int = 0,
    ) -> List[StoredAssessment]:
        query: Dict[str, Any] = {}
        window = _build_date_window(date, from_date, to_date)
        if window:
            query["created_at"] = window

        cursor = (
            self._collection.find(query)
            .sort("created_at", -1)
            .skip(skip)
            .limit(limit)
        )
        return [_to_stored(doc) async for doc in cursor]


def _build_date_window(
    date: Optional[str], from_date: Optional[str], to_date: Optional[str]
) -> Dict[str, datetime]:
    """Translate date filters into a Mongo range query on created_at (UTC)."""
    window: Dict[str, datetime] = {}
    if date:
        day = datetime.strptime(date, "%Y-%m-%d").date()
        window["$gte"] = datetime.combine(day, time.min, tzinfo=timezone.utc)
        window["$lt"] = datetime.combine(day, time.max, tzinfo=timezone.utc)
        return window
    if from_date:
        day = datetime.strptime(from_date, "%Y-%m-%d").date()
        window["$gte"] = datetime.combine(day, time.min, tzinfo=timezone.utc)
    if to_date:
        day = datetime.strptime(to_date, "%Y-%m-%d").date()
        window["$lt"] = datetime.combine(day, time.max, tzinfo=timezone.utc)
    return window
