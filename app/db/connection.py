"""MongoDB connection management (Motor async client)."""
from __future__ import annotations

import logging
from typing import Optional

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

from app.config import get_settings

logger = logging.getLogger(__name__)

_client: Optional[AsyncIOMotorClient] = None


async def connect() -> AsyncIOMotorClient:
    global _client
    if _client is None:
        settings = get_settings()
        logger.info("Connecting to MongoDB database '%s'", settings.database_name)
        _client = AsyncIOMotorClient(
            settings.mongodb_uri,
            serverSelectionTimeoutMS=5000,
        )
        await _client.admin.command("ping")
        await get_database().get_collection(
            settings.assessments_collection
        ).create_index("created_at")
    return _client


async def disconnect() -> None:
    global _client
    if _client is not None:
        _client.close()
        _client = None
        logger.info("MongoDB connection closed")


def get_database() -> AsyncIOMotorDatabase:
    if _client is None:
        raise RuntimeError("MongoDB is not connected. Call connect() first.")
    return _client[get_settings().database_name]