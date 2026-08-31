import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from bson import ObjectId
from bson.errors import InvalidId
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorCollection
from pymongo.errors import PyMongoError

from app.config import Settings
from app.errors import PipelineError
from app.models import FirstAssessment

logger = logging.getLogger(__name__)

_client: Optional[AsyncIOMotorClient] = None


def get_collection(settings: Settings) -> AsyncIOMotorCollection:
    global _client
    if _client is None:
        _client = AsyncIOMotorClient(
            settings.mongodb_uri, serverSelectionTimeoutMS=5000, tz_aware=True
        )
    return _client[settings.mongodb_database][settings.mongodb_collection]


def close_client() -> None:
    global _client
    if _client is not None:
        _client.close()
        _client = None


class AssessmentStore:
    def __init__(self, collection: AsyncIOMotorCollection):
        self.collection = collection

    async def save(
        self, assessment: FirstAssessment, metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        document = {
            "createdAt": datetime.now(timezone.utc),
            "assessment": assessment.model_dump(),
            "metadata": metadata or {},
        }
        try:
            result = await self.collection.insert_one(document)
        except PyMongoError as exc:
            raise _database_error("Could not save the assessment.", exc) from exc
        document["_id"] = result.inserted_id
        return _record(document)

    async def get(self, assessment_id: str) -> Dict[str, Any]:
        try:
            object_id = ObjectId(assessment_id)
        except (InvalidId, TypeError) as exc:
            raise _not_found(assessment_id, "not a valid assessment id") from exc
        try:
            document = await self.collection.find_one({"_id": object_id})
        except PyMongoError as exc:
            raise _database_error("Could not read the assessment.", exc) from exc
        if document is None:
            raise _not_found(assessment_id, "no assessment with this id")
        return _record(document)

    async def list(
        self,
        created_from: Optional[datetime] = None,
        created_to: Optional[datetime] = None,
        limit: int = 50,
        skip: int = 0,
    ) -> List[Dict[str, Any]]:
        created: Dict[str, datetime] = {}
        if created_from is not None:
            created["$gte"] = created_from
        if created_to is not None:
            created["$lte"] = created_to
        query = {"createdAt": created} if created else {}
        try:
            cursor = (
                self.collection.find(query).sort("createdAt", -1).skip(skip).limit(limit)
            )
            documents = await cursor.to_list(length=limit)
        except PyMongoError as exc:
            raise _database_error("Could not list assessments.", exc) from exc
        return [_record(document) for document in documents]


def _record(document: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": str(document["_id"]),
        "createdAt": document["createdAt"],
        "assessment": document.get("assessment", {}),
        "metadata": document.get("metadata", {}),
    }


def _database_error(message: str, exc: Exception) -> PipelineError:
    logger.warning("mongodb error: %s", exc)
    return PipelineError(
        "database_unavailable",
        message,
        503,
        [{"field": "database", "message": str(exc)}],
    )


def _not_found(assessment_id: str, reason: str) -> PipelineError:
    return PipelineError(
        "assessment_not_found",
        f"Assessment '{assessment_id}' was not found.",
        404,
        [{"field": "id", "message": reason}],
    )
