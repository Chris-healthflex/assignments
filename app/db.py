from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorCollection

from app.config import get_settings

_client: AsyncIOMotorClient | None = None


def get_client() -> AsyncIOMotorClient:
    global _client
    if _client is None:
        settings = get_settings()
        _client = AsyncIOMotorClient(settings.mongo_uri)
    return _client


def get_collection() -> AsyncIOMotorCollection:
    settings = get_settings()
    return get_client()[settings.mongo_db_name][settings.mongo_collection]
