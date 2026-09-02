from functools import lru_cache

from pymongo import MongoClient
from pymongo.database import Database

from app.config import get_settings


@lru_cache
def get_mongo_client() -> MongoClient:
    settings = get_settings()

    client = MongoClient(
        settings.mongodb_uri,
        serverSelectionTimeoutMS=5000,
    )

    # Force connection check during startup/use.
    client.admin.command("ping")

    return client


def get_database() -> Database:
    settings = get_settings()
    client = get_mongo_client()

    return client[settings.mongodb_database]


def close_mongo_client() -> None:
    try:
        client = get_mongo_client()
        client.close()
        get_mongo_client.cache_clear()
    except Exception:
        pass
