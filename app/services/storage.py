"""MongoDB persistence for saved assessments."""

import asyncio
import logging
from datetime import datetime, timezone
from typing import List, Optional

from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorClient

from app.config import COLLECTION_NAME, DB_NAME, MONGODB_URI

logger = logging.getLogger(__name__)

# Motor binds a client to the event loop it is first used on, so the client is
# cached per loop rather than created per call. Under uvicorn there is one loop
# for the process, so this is a single client with a single connection pool;
# TestClient spins up a fresh loop per request, so each gets its own.
_client = None
_client_loop = None


def get_mongo_collection():
    """Return the Motor async collection, reusing the client for the active loop."""
    global _client, _client_loop

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if _client is None or _client_loop is not loop:
        _client = AsyncIOMotorClient(MONGODB_URI)
        _client_loop = loop

    return _client[DB_NAME][COLLECTION_NAME]


async def save_assessment_to_db(assessment_dict: dict) -> dict:
    """Save a parsed FirstAssessment payload into the assessments collection."""
    collection = get_mongo_collection()
    now_utc = datetime.now(timezone.utc)

    doc = {
        "createdAt": now_utc.isoformat(),
        "date": now_utc.strftime("%Y-%m-%d"),
        "assessment": assessment_dict
    }
    result = await collection.insert_one(doc)
    return {
        "id": str(result.inserted_id),
        "createdAt": doc["createdAt"],
        "date": doc["date"],
        "assessment": assessment_dict
    }


async def get_assessment_by_id(doc_id: str) -> Optional[dict]:
    """Retrieve an assessment document by its unique ID."""
    collection = get_mongo_collection()
    try:
        query = {"_id": ObjectId(doc_id)}
    except Exception:
        query = {"_id": doc_id}

    doc = await collection.find_one(query)
    if doc:
        doc["id"] = str(doc["_id"])
        del doc["_id"]
        return doc
    return None


async def list_assessments(filter_date: Optional[str] = None) -> List[dict]:
    """List saved assessment documents, optionally filtered by date (YYYY-MM-DD)."""
    collection = get_mongo_collection()
    query = {}
    if filter_date:
        query["$or"] = [
            {"date": filter_date},
            {"createdAt": {"$regex": f"^{filter_date}"}}
        ]

    results = []
    async for doc in collection.find(query):
        doc["id"] = str(doc["_id"])
        del doc["_id"]
        results.append(doc)
    return results
