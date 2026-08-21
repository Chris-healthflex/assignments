from typing import List, Optional
from datetime import datetime, date
from motor.motor_asyncio import AsyncIOMotorDatabase
from bson import ObjectId

from app.db.client import get_database
from app.db.models import AssessmentDocument
from app.schemas.assessment import FirstAssessment

async def save_assessment(assessment: FirstAssessment, transcript: str | None = None) -> str:
    """Save assessment and return inserted id."""
    db = get_database()
    doc = AssessmentDocument(
        assessment=assessment.model_dump(),
        transcript=transcript
    )
    result = await db.assessments.insert_one(doc.model_dump(by_alias=True, exclude={"id"}))
    return str(result.inserted_id)

async def get_assessment_by_id(id: str) -> Optional[dict]:
    db = get_database()
    try:
        oid = ObjectId(id)
    except:
        return None
    doc = await db.assessments.find_one({"_id": oid})
    if doc:
        doc["_id"] = str(doc["_id"])
    return doc

async def list_assessments(start_date: Optional[date] = None, end_date: Optional[date] = None) -> List[dict]:
    db = get_database()
    query = {}
    if start_date:
        query["created_at"] = query.get("created_at", {})
        query["created_at"]["$gte"] = datetime.combine(start_date, datetime.min.time())
    if end_date:
        query["created_at"] = query.get("created_at", {})
        query["created_at"]["$lte"] = datetime.combine(end_date, datetime.max.time())
    cursor = db.assessments.find(query).sort("created_at", -1)
    results = []
    async for doc in cursor:
        doc["_id"] = str(doc["_id"])
        results.append(doc)
    return results