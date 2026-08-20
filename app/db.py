"""MongoDB connection plus save/retrieve for FirstAssessment documents."""

from __future__ import annotations

import logging
from typing import Any

from bson import ObjectId
from bson.errors import InvalidId
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorCollection

from app.config import get_settings
from app.schemas import FirstAssessment

logger = logging.getLogger(__name__)

_client: AsyncIOMotorClient | None = None


def get_client() -> AsyncIOMotorClient:
    """Lazily create the process-wide Motor client."""
    global _client
    if _client is None:
        settings = get_settings()
        _client = AsyncIOMotorClient(settings.mongodb_uri, tz_aware=True)
        logger.info("Mongo client created for %s", settings.mongodb_db)
    return _client


def get_collection() -> AsyncIOMotorCollection:
    settings = get_settings()
    return get_client()[settings.mongodb_db][settings.mongodb_assessments_collection]


async def ping() -> bool:
    """Cheap liveness check used by /health."""
    try:
        await get_client().admin.command("ping")
        return True
    except Exception:  # noqa: BLE001 -- health check must never raise
        logger.exception("Mongo ping failed")
        return False


async def ensure_indexes() -> None:
    """Idempotent index creation; called once on startup."""
    coll = get_collection()
    await coll.create_index("session_id")
    await coll.create_index("created_at")


async def save_assessment(assessment: FirstAssessment) -> str:
    """Insert an assessment and return its id as a string."""
    doc: dict[str, Any] = assessment.model_dump(mode="python", exclude={"assessment_id"})
    result = await get_collection().insert_one(doc)
    return str(result.inserted_id)


async def get_assessment(assessment_id: str) -> FirstAssessment | None:
    """Fetch one assessment by id. Returns None for missing or malformed ids."""
    try:
        oid = ObjectId(assessment_id)
    except (InvalidId, TypeError):
        return None
    doc = await get_collection().find_one({"_id": oid})
    return _to_model(doc) if doc else None


async def list_assessments(limit: int = 20) -> list[FirstAssessment]:
    cursor = get_collection().find().sort("created_at", -1).limit(limit)
    return [_to_model(doc) async for doc in cursor]


def _to_model(doc: dict[str, Any]) -> FirstAssessment:
    doc = dict(doc)
    doc["assessment_id"] = str(doc.pop("_id"))
    return FirstAssessment.model_validate(doc)


async def close() -> None:
    global _client
    if _client is not None:
        _client.close()
        _client = None
