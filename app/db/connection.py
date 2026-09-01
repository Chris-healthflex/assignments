from pymongo import MongoClient
from app.core.config import MONGO_URI, MONGO_DB_NAME

_client = None


def get_db():
    """
    Returns the MongoDB database instance, reusing a single client connection.
    """
    global _client
    if _client is None:
        _client = MongoClient(MONGO_URI)
    return _client[MONGO_DB_NAME]