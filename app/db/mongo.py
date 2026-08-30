from __future__ import annotations

from datetime import date, datetime, time, timezone
from typing import Any

from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorCollection
from pymongo.errors import PyMongoError, ServerSelectionTimeoutError

from app.config import Settings, get_settings
from app.errors import BadRequestError, DatabaseError, DatabaseUnavailableError
from app.schemas.first_assessment import FirstAssessment

_client: AsyncIOMotorClient | None = None
_collection: AsyncIOMotorCollection | None = None


async def connect_mongo(settings: Settings | None = None) -> None:
    global _client, _collection

    settings = settings or get_settings()
    _client = AsyncIOMotorClient(settings.mongodb_uri, serverSelectionTimeoutMS=3000)
    _collection = _client[settings.mongodb_database][settings.mongodb_collection]

    try:
        await _client.admin.command("ping")
    except ServerSelectionTimeoutError as exc:
        raise DatabaseUnavailableError("MongoDB is unavailable") from exc


async def close_mongo() -> None:
    global _client, _collection

    if _client is not None:
        _client.close()
    _client = None
    _collection = None


def get_collection(settings: Settings | None = None) -> AsyncIOMotorCollection:
    global _client, _collection

    if _collection is None:
        settings = settings or get_settings()
        _client = AsyncIOMotorClient(settings.mongodb_uri, serverSelectionTimeoutMS=3000)
        _collection = _client[settings.mongodb_database][settings.mongodb_collection]

    return _collection


def serialize_assessment_doc(document: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": str(document["_id"]),
        "createdAt": document["createdAt"].isoformat(),
        "assessment": document["assessment"],
        **({"filename": document["filename"]} if document.get("filename") else {}),
    }


async def save_assessment(
    assessment: FirstAssessment,
    *,
    filename: str | None = None,
    confidence: dict[str, float] | None = None,
) -> dict[str, Any]:
    document: dict[str, Any] = {
        "createdAt": datetime.now(timezone.utc),
        "assessment": assessment.model_dump(),
    }

    if filename:
        document["filename"] = filename
    if confidence is not None:
        document["confidence"] = confidence

    try:
        result = await get_collection().insert_one(document)
    except ServerSelectionTimeoutError as exc:
        raise DatabaseUnavailableError("MongoDB is unavailable") from exc
    except PyMongoError as exc:
        raise DatabaseError("Could not save assessment") from exc

    document["_id"] = result.inserted_id
    return serialize_assessment_doc(document)


async def get_assessment_by_id(assessment_id: str) -> dict[str, Any] | None:
    if not ObjectId.is_valid(assessment_id):
        raise BadRequestError("Invalid assessment id")

    try:
        document = await get_collection().find_one({"_id": ObjectId(assessment_id)})
    except ServerSelectionTimeoutError as exc:
        raise DatabaseUnavailableError("MongoDB is unavailable") from exc
    except PyMongoError as exc:
        raise DatabaseError("Could not retrieve assessment") from exc

    return serialize_assessment_doc(document) if document else None


async def list_assessments(created_on: date | None = None) -> list[dict[str, Any]]:
    query: dict[str, Any] = {}
    if created_on is not None:
        start = datetime.combine(created_on, time.min, tzinfo=timezone.utc)
        end = datetime.combine(created_on, time.max, tzinfo=timezone.utc)
        query["createdAt"] = {"$gte": start, "$lte": end}

    try:
        cursor = get_collection().find(query).sort("createdAt", -1)
        documents = await cursor.to_list(length=None)
    except ServerSelectionTimeoutError as exc:
        raise DatabaseUnavailableError("MongoDB is unavailable") from exc
    except PyMongoError as exc:
        raise DatabaseError("Could not list assessments") from exc

    return [serialize_assessment_doc(document) for document in documents]
