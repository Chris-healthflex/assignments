from __future__ import annotations

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

from app.core.config import get_settings

_client: AsyncIOMotorClient | None = None
_db: AsyncIOMotorDatabase | None = None


def connect_to_mongo() -> None:
    """Create the Mongo client. Call once on FastAPI startup."""
    global _client, _db
    settings = get_settings()
    _client = AsyncIOMotorClient(settings.mongodb_uri)
    _db = _client[settings.mongodb_db_name]


def close_mongo_connection() -> None:
    """Close the Mongo client. Call once on FastAPI shutdown."""
    global _client, _db
    if _client is not None:
        _client.close()
    _client = None
    _db = None


def get_database() -> AsyncIOMotorDatabase:
    if _db is None:
        # Lazy-connect fallback (useful for scripts/tests that don't run
        # the FastAPI lifespan hooks).
        connect_to_mongo()
    assert _db is not None
    return _db


def get_assessments_collection():
    return get_database()["assessments"]
