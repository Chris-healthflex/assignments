from datetime import datetime, timezone
from bson import ObjectId
from app.db.connection import get_db
from app.schemas.first_assessment import FirstAssessment

COLLECTION_NAME = "assessments"


def save_assessment(assessment: FirstAssessment) -> str:
    """
    Saves a FirstAssessment to MongoDB. Returns the inserted document's ID as a string.
    """
    db = get_db()
    doc = assessment.model_dump()
    doc["createdAt"] = datetime.now(timezone.utc)

    result = db[COLLECTION_NAME].insert_one(doc)
    return str(result.inserted_id)


def get_assessment_by_id(assessment_id: str) -> dict | None:
    """
    Retrieves a single assessment by its MongoDB ObjectId string.
    Returns None if not found or if the ID is invalid.
    """
    db = get_db()
    try:
        oid = ObjectId(assessment_id)
    except Exception:
        return None

    doc = db[COLLECTION_NAME].find_one({"_id": oid})
    if doc:
        doc["_id"] = str(doc["_id"])
    return doc


def list_assessments(date_filter: str | None = None) -> list[dict]:
    """
    Lists all assessments, optionally filtered by a date (YYYY-MM-DD, matches createdAt day).
    """
    db = get_db()
    query = {}

    if date_filter:
        start = datetime.strptime(date_filter, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        end = start.replace(hour=23, minute=59, second=59)
        query["createdAt"] = {"$gte": start, "$lte": end}

    docs = list(db[COLLECTION_NAME].find(query).sort("createdAt", -1))
    for doc in docs:
        doc["_id"] = str(doc["_id"])
    return docs