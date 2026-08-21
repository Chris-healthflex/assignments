from datetime import datetime, timezone

from bson import ObjectId
from bson.errors import InvalidId
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorCollection

from app.schemas.first_assessment import FirstAssessment

COLLECTION_NAME = "assessments"


class AssessmentNotFoundError(Exception):
    pass


class AssessmentRepository:
    def __init__(self, collection: AsyncIOMotorCollection):
        self._collection = collection

    async def save(self, assessment: FirstAssessment) -> str:
        document = assessment.model_dump()
        document["createdAt"] = datetime.now(timezone.utc)
        result = await self._collection.insert_one(document)
        return str(result.inserted_id)

    async def get(self, assessment_id: str) -> dict:
        try:
            object_id = ObjectId(assessment_id)
        except InvalidId as exc:
            raise AssessmentNotFoundError(assessment_id) from exc

        document = await self._collection.find_one({"_id": object_id})
        if document is None:
            raise AssessmentNotFoundError(assessment_id)

        return _serialize(document)

    async def list(
        self, date_from: datetime | None = None, date_to: datetime | None = None
    ) -> list[dict]:
        query: dict = {}
        date_filter = {}
        if date_from is not None:
            date_filter["$gte"] = date_from
        if date_to is not None:
            date_filter["$lte"] = date_to
        if date_filter:
            query["createdAt"] = date_filter

        cursor = self._collection.find(query).sort("createdAt", -1)
        return [_serialize(doc) async for doc in cursor]


def _serialize(document: dict) -> dict:
    document = dict(document)
    document["id"] = str(document.pop("_id"))
    if isinstance(document.get("createdAt"), datetime):
        document["createdAt"] = document["createdAt"].isoformat()
    return document


def get_repository(client: AsyncIOMotorClient, db_name: str) -> AssessmentRepository:
    collection = client[db_name][COLLECTION_NAME]
    return AssessmentRepository(collection)
