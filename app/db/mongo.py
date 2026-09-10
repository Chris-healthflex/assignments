from typing import Optional
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from app.config import settings

_client: Optional[AsyncIOMotorClient] = None


def get_client() -> AsyncIOMotorClient:
    global _client
    if _client is None:
        _client = AsyncIOMotorClient(settings.MONGODB_URL)
    return _client


def get_database() -> AsyncIOMotorDatabase:
    client = get_client()
    return client[settings.DATABASE_NAME]


async def init_db() -> None:
    """Initializes database indexes."""
    db = get_database()
    collection = db[settings.COLLECTION_NAME]
    # Ensure descending index on createdAt for fast sorting and date-range queries
    await collection.create_index([("createdAt", -1)])


async def close_db() -> None:
    """Closes the MongoDB client connection."""
    global _client
    if _client is not None:
        _client.close()
        _client = None
