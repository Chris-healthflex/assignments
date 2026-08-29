from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from bson import ObjectId
from bson.errors import InvalidId

from app.db.mongodb import get_assessments_collection
from app.schemas.assessment import FirstAssessment


class AssessmentRepository:
    """All MongoDB access for assessments lives here — nowhere else in the
    codebase should import pymongo/motor or touch the collection directly."""

    @staticmethod
    async def save(assessment: FirstAssessment) -> str:
        collection = get_assessments_collection()
        document: dict[str, Any] = assessment.model_dump()
        document["createdAt"] = datetime.now(timezone.utc)
        result = await collection.insert_one(document)
        return str(result.inserted_id)

    @staticmethod
    async def get_by_id(assessment_id: str) -> dict[str, Any] | None:
        collection = get_assessments_collection()
        try:
            object_id = ObjectId(assessment_id)
        except (InvalidId, TypeError):
            return None
        document = await collection.find_one({"_id": object_id})
        if document is None:
            return None
        return _serialize(document)

    @staticmethod
    async def list_all(
        date_from: datetime | None = None,
        date_to: datetime | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        collection = get_assessments_collection()
        query: dict[str, Any] = {}
        if date_from or date_to:
            date_filter: dict[str, Any] = {}
            if date_from:
                date_filter["$gte"] = date_from
            if date_to:
                date_filter["$lte"] = date_to
            query["createdAt"] = date_filter

        cursor = collection.find(query).sort("createdAt", -1).limit(limit)
        documents = await cursor.to_list(length=limit)
        return [_serialize(document) for document in documents]


def _serialize(document: dict[str, Any]) -> dict[str, Any]:
    """Convert Mongo's ObjectId/datetime into JSON-friendly values and
    rename `_id` -> `id` for the API response."""
    document = dict(document)
    document["id"] = str(document.pop("_id"))
    if isinstance(document.get("createdAt"), datetime):
        document["createdAt"] = document["createdAt"].isoformat()
    return document
