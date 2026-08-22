import datetime
import logging
from typing import Optional, List, Dict, Any
from bson import ObjectId
from bson.errors import InvalidId
from pymongo import MongoClient
from pymongo.server_api import ServerApi
from pymongo.errors import ConnectionFailure, PyMongoError

from app.core.config import settings

logger = logging.getLogger(__name__)

# Global client holder (singleton per lifespan)
_mongo_client: Optional[MongoClient] = None

def init_db() -> bool:
    """
    Initializes the MongoClient and returns True if connection ping is successful.
    Even if ping fails, client is kept initialized so we can try to recover or fail later.
    """
    global _mongo_client
    if not settings.mongodb_uri:
        logger.error("MONGODB_URI is not set in settings!")
        return False
    try:
        _mongo_client = MongoClient(settings.mongodb_uri, server_api=ServerApi("1"))
        # Force a connection check
        _mongo_client.admin.command("ping")
        logger.info("MongoDB Atlas connected successfully.")
        
        # Ensure indexes
        db = _mongo_client[settings.mongodb_database]
        collection = db["assessments"]
        collection.create_index("_id")
        collection.create_index("created_at")
        logger.info("MongoDB indexes verified.")
        return True
    except (ConnectionFailure, PyMongoError) as e:
        logger.error(f"Failed to connect to MongoDB Atlas at startup: {e}")
        return False

def close_db():
    global _mongo_client
    if _mongo_client:
        _mongo_client.close()
        _mongo_client = None
        logger.info("MongoDB connection closed.")

def get_database():
    if _mongo_client is None:
        raise ConnectionError("Database client not initialized")
    return _mongo_client[settings.mongodb_database]

def get_collection():
    db = get_database()
    return db["assessments"]

def check_db_health() -> bool:
    if _mongo_client is None:
        return False
    try:
        _mongo_client.admin.command("ping")
        return True
    except Exception:
        return False

# --- Repository Layer ---

def create_assessment(assessment_data: dict) -> str:
    """
    Persists the assessment data to the collection, appending DB metadata
    (created_at, updated_at). Returns the inserted document ID as a string.
    """
    try:
        collection = get_collection()
        doc = assessment_data.copy()
        # Add metadata
        now = datetime.datetime.now(datetime.timezone.utc)
        doc["created_at"] = now
        doc["updated_at"] = now
        
        result = collection.insert_one(doc)
        return str(result.inserted_id)
    except (ConnectionFailure, PyMongoError) as e:
        logger.error(f"Database insertion failed: {e}")
        raise ConnectionError("Database write error") from e

def get_assessment_by_id(assessment_id: str) -> Optional[dict]:
    """
    Retrieves an assessment document by its ObjectId string.
    If the format is invalid, raises ValueError.
    If the document is missing, returns None.
    """
    try:
        oid = ObjectId(assessment_id)
    except InvalidId as e:
        logger.warning(f"Invalid ObjectId format: {assessment_id}")
        raise ValueError("Invalid ID format") from e

    try:
        collection = get_collection()
        doc = collection.find_one({"_id": oid})
        return doc
    except (ConnectionFailure, PyMongoError) as e:
        logger.error(f"Database query failed for ID {assessment_id}: {e}")
        raise ConnectionError("Database read error") from e

def list_assessments(
    limit: int = 20,
    offset: int = 0,
    start_date: Optional[datetime.datetime] = None,
    end_date: Optional[datetime.datetime] = None
) -> List[dict]:
    """
    Lists assessments from the database, sorted by created_at in descending order.
    Optionally filters by start_date and end_date.
    """
    try:
        collection = get_collection()
        query: Dict[str, Any] = {}
        
        # Date filtering on created_at metadata
        date_filter: Dict[str, Any] = {}
        if start_date:
            date_filter["$gte"] = start_date
        if end_date:
            date_filter["$lte"] = end_date
            
        if date_filter:
            query["created_at"] = date_filter

        cursor = collection.find(query).sort("created_at", -1).skip(offset).limit(limit)
        return list(cursor)
    except (ConnectionFailure, PyMongoError) as e:
        logger.error(f"Database list query failed: {e}")
        raise ConnectionError("Database read error") from e
