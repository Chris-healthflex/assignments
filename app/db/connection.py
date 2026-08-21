from functools import lru_cache

from pymongo import MongoClient

from app.core.config import get_settings


@lru_cache
def get_client() -> MongoClient:
    return MongoClient(get_settings().mongodb_uri)


def get_collection():
    settings = get_settings()
    return get_client()[settings.mongodb_database][settings.mongodb_collection]
