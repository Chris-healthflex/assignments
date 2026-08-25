"""MongoDB connection and CRUD helpers."""
from __future__ import annotations

import os
from datetime import date, datetime, timezone
from typing import Any

from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorCollection

from .schemas import FirstAssessment, SavedAssessment


class DatabaseError(RuntimeError):
    pass


class AssessmentRepository:
    def __init__(self, collection: AsyncIOMotorCollection):
        self.collection = collection

    @staticmethod
    async def from_environment() -> "AssessmentRepository":
        uri = os.getenv("MONGODB_URI")
        if not uri:
            raise DatabaseError("MONGODB_URI is not configured.")
        try:
            client = AsyncIOMotorClient(uri, serverSelectionTimeoutMS=3000)
            await client.admin.command("ping")
            database = client[os.getenv("MONGODB_DATABASE", "clinical_assessments")]
            return AssessmentRepository(database.assessments)
        except Exception as exc:
            raise DatabaseError(f"MongoDB connection failed: {exc}") from exc

    @staticmethod
    def _serialize(document: dict[str, Any]) -> SavedAssessment:
        return SavedAssessment(
            **{key: value for key, value in document.items() if key not in {"_id", "createdAt"}},
            id=str(document["_id"]),
            createdAt=document["createdAt"].isoformat(),
        )

    async def create(self, assessment: FirstAssessment) -> SavedAssessment:
        document = assessment.model_dump(mode="json")
        document["createdAt"] = datetime.now(timezone.utc)
        result = await self.collection.insert_one(document)
        document["_id"] = result.inserted_id
        return self._serialize(document)

    async def get(self, assessment_id: str) -> SavedAssessment | None:
        if not ObjectId.is_valid(assessment_id):
            return None
        document = await self.collection.find_one({"_id": ObjectId(assessment_id)})
        return self._serialize(document) if document else None

    async def list(self, created_date: date | None = None) -> list[SavedAssessment]:
        query: dict[str, Any] = {}
        if created_date:
            start = datetime.combine(created_date, datetime.min.time(), tzinfo=timezone.utc)
            end = datetime.combine(created_date, datetime.max.time(), tzinfo=timezone.utc)
            query["createdAt"] = {"$gte": start, "$lte": end}
        documents = await self.collection.find(query).sort("createdAt", -1).to_list(length=100)
        return [self._serialize(document) for document in documents]
