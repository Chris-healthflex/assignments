from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Optional

from bson import ObjectId
from bson.errors import InvalidId
from pymongo import AsyncMongoClient, DESCENDING
from pymongo.errors import PyMongoError

from app.config import Settings
from app.schemas.api import ExtractionMeta, StoredAssessment
from app.schemas.assessment import FirstAssessment

logger = logging.getLogger(__name__)


class StorageError(RuntimeError):
    """Raised when the database is unreachable or rejects an operation."""


class InvalidAssessmentId(ValueError):
    """Raised when a caller-supplied id is not a valid ObjectId."""


class AssessmentRepository:
    """Thin data-access layer around one collection."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client: AsyncMongoClient | None = None

    async def connect(self) -> None:
        self._client = AsyncMongoClient(
            self._settings.mongodb_uri,
            serverSelectionTimeoutMS=self._settings.mongodb_timeout_ms,
            tz_aware=True,
        )
        try:
            await self._collection.create_index([("createdAt", DESCENDING)])
        except PyMongoError as exc:
            logger.warning("Could not ensure createdAt index: %s", exc)

    async def close(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None

    @property
    def _collection(self):
        if self._client is None:
            raise StorageError("Repository used before connect() was called.")
        return self._client[self._settings.mongodb_db][self._settings.mongodb_collection]

    async def ping(self) -> bool:
        try:
            await self._client.admin.command("ping") 
            return True
        except (PyMongoError, AttributeError) as exc:
            logger.warning("Mongo ping failed: %s", exc)
            return False

    async def save(
        self, assessment: FirstAssessment, meta: ExtractionMeta | None = None
    ) -> StoredAssessment:
        doc: dict[str, Any] = {
            "createdAt": datetime.now(timezone.utc),
            "assessment": assessment.model_dump(),
            "meta": meta.model_dump() if meta else None,
        }
        try:
            result = await self._collection.insert_one(doc)
        except PyMongoError as exc:
            raise StorageError(f"Failed to save assessment: {exc}") from exc

        return StoredAssessment(
            id=str(result.inserted_id),
            createdAt=doc["createdAt"],
            assessment=assessment,
            meta=meta,
        )

    async def get(self, assessment_id: str) -> Optional[StoredAssessment]:
        try:
            oid = ObjectId(assessment_id)
        except (InvalidId, TypeError) as exc:
            raise InvalidAssessmentId(str(exc)) from exc

        try:
            doc = await self._collection.find_one({"_id": oid})
        except PyMongoError as exc:
            raise StorageError(f"Failed to read assessment: {exc}") from exc

        return self._to_model(doc) if doc else None

    async def list(
        self,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
        limit: int = 50,
        skip: int = 0,
    ) -> tuple[int, list[StoredAssessment]]:
        query: dict[str, Any] = {}
        if date_from or date_to:
            window: dict[str, datetime] = {}
            if date_from:
                window["$gte"] = date_from
            if date_to:
                window["$lte"] = date_to
            query["createdAt"] = window

        try:
            total = await self._collection.count_documents(query)
            cursor = (
                self._collection.find(query)
                .sort("createdAt", DESCENDING)
                .skip(skip)
                .limit(limit)
            )
            docs = [d async for d in cursor]
        except PyMongoError as exc:
            raise StorageError(f"Failed to list assessments: {exc}") from exc

        return total, [self._to_model(d) for d in docs]

    @staticmethod
    def _to_model(doc: dict[str, Any]) -> StoredAssessment:
        created = doc.get("createdAt")
        if created is not None and created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        return StoredAssessment(
            id=str(doc["_id"]),
            createdAt=created,
            assessment=FirstAssessment.model_validate(doc.get("assessment") or {}),
            meta=(
                ExtractionMeta.model_validate(doc["meta"])
                if doc.get("meta")
                else None
            ),
        )
