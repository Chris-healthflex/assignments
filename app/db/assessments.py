from datetime import datetime, time, timedelta

from bson import ObjectId

from app.db.connection import get_collection
from app.models.api_models import AssessmentRecord
from app.models.first_assessment import FirstAssessment


def _to_record(document: dict) -> AssessmentRecord:
    payload = {key: value for key, value in document.items() if key not in {"_id", "created_at"}}
    return AssessmentRecord(
        id=str(document["_id"]),
        created_at=document["created_at"],
        **payload,
    )


def save_assessment(assessment: FirstAssessment) -> AssessmentRecord:
    now = datetime.utcnow()
    document = assessment.model_dump(mode="json")
    document["created_at"] = now
    result = get_collection().insert_one(document)
    document["_id"] = result.inserted_id
    return _to_record(document)


def get_assessment(assessment_id: str) -> AssessmentRecord | None:
    if not ObjectId.is_valid(assessment_id):
        return None
    document = get_collection().find_one({"_id": ObjectId(assessment_id)})
    return _to_record(document) if document else None


def _end_of_day(value: datetime) -> datetime:
    """If to_date has no explicit time component (a bare date, e.g. 2026-08-21),
    treat it as inclusive of the whole day rather than as midnight. A record
    created at 05:50 on that date must still match a to_date filter for that
    same date.
    """
    if value.time() == time.min:
        return value + timedelta(days=1) - timedelta(microseconds=1)
    return value


def list_assessments(from_date: datetime | None = None, to_date: datetime | None = None) -> list[AssessmentRecord]:
    query: dict = {}
    if from_date or to_date:
        query["created_at"] = {}
        if from_date:
            query["created_at"]["$gte"] = from_date
        if to_date:
            query["created_at"]["$lte"] = _end_of_day(to_date)
    return [_to_record(document) for document in get_collection().find(query).sort("created_at", -1)]
