import os
from datetime import datetime, timezone
from typing import Any

from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()


_client: MongoClient | None = None


def get_database():
    global _client

    mongodb_uri = os.getenv("MONGODB_URI", "").strip()

    if not mongodb_uri:
        raise RuntimeError(
            "MONGODB_URI is not configured. Add your MongoDB connection "
            "string to the local .env file."
        )

    database_name = os.getenv("MONGODB_DATABASE", "stance_health")

    if _client is None:
        _client = MongoClient(
            mongodb_uri,
            serverSelectionTimeoutMS=5000,
        )

    return _client[database_name]


def get_assessments_collection():
    database = get_database()
    return database["assessments"]


def save_assessment(assessment: dict[str, Any]) -> str:
    collection = get_assessments_collection()

    document = {
        "assessment": assessment,
        "createdAt": datetime.now(timezone.utc),
    }

    result = collection.insert_one(document)

    return str(result.inserted_id)


def get_assessment(assessment_id: str) -> dict[str, Any] | None:
    from bson import ObjectId

    collection = get_assessments_collection()

    try:
        object_id = ObjectId(assessment_id)
    except Exception:
        return None

    document = collection.find_one({"_id": object_id})

    if document is None:
        return None

    return {
        "id": str(document["_id"]),
        "assessment": document["assessment"],
        "createdAt": document["createdAt"].isoformat(),
    }


def list_assessments(
    start_date: datetime | None = None,
    end_date: datetime | None = None,
) -> list[dict[str, Any]]:
    collection = get_assessments_collection()

    query: dict[str, Any] = {}

    if start_date or end_date:
        date_filter: dict[str, Any] = {}

        if start_date:
            date_filter["$gte"] = start_date

        if end_date:
            date_filter["$lt"] = end_date

        query["createdAt"] = date_filter

    documents = collection.find(query).sort("createdAt", -1)

    return [
        {
            "id": str(document["_id"]),
            "assessment": document["assessment"],
            "createdAt": document["createdAt"].isoformat(),
        }
        for document in documents
    ]