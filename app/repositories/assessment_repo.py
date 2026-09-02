"""Async Assessment Repository for MongoDB persistence."""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union
from bson import ObjectId
from bson.errors import InvalidId
from pymongo.errors import PyMongoError

from app.db.models import AssessmentDocument
from app.db.mongo import MongoDBManager, db_manager
from app.schemas.assessment import FirstAssessment


class RepositoryException(Exception):
    """Base exception for repository operations."""

    pass


class InvalidAssessmentIdError(RepositoryException):
    """Raised when an invalid ObjectId format is provided."""

    pass


class AssessmentNotFoundError(RepositoryException):
    """Raised when an assessment document is not found."""

    pass


class AssessmentRepository:
    """Repository managing asynchronous persistence of FirstAssessment documents."""

    def __init__(self, manager: Optional[MongoDBManager] = None) -> None:
        """Initialize repository with MongoDBManager.

        Args:
            manager: Optional MongoDBManager override (defaults to global db_manager).
        """
        self.manager = manager or db_manager

    @property
    def collection(self):
        """Get the underlying async Motor collection."""
        return self.manager.get_collection()

    async def ensure_indexes(self) -> None:
        """Create required indexes on the collection (e.g. created_at for date filtering)."""
        try:
            await self.collection.create_index([("created_at", -1)], name="idx_created_at_desc")
        except PyMongoError as exc:
            raise RepositoryException(f"Failed to create indexes: {str(exc)}") from exc

    def _parse_datetime(self, date_val: Union[datetime, str]) -> datetime:
        """Safely parse a string or datetime into a timezone-aware UTC datetime.

        Args:
            date_val: datetime object or ISO-8601 date string.

        Returns:
            Timezone-aware datetime in UTC.

        Raises:
            ValueError: If date string cannot be parsed.
        """
        if isinstance(date_val, datetime):
            if date_val.tzinfo is None:
                return date_val.replace(tzinfo=timezone.utc)
            return date_val.astimezone(timezone.utc)

        if isinstance(date_val, str):
            # Parse ISO-8601 string
            clean_str = date_val.strip()
            # Handle 'Z' suffix
            if clean_str.endswith("Z"):
                clean_str = clean_str[:-1] + "+00:00"
            dt = datetime.fromisoformat(clean_str)
            if dt.tzinfo is None:
                return dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)

        raise ValueError(f"Invalid date value type: {type(date_val)}")

    async def create(
        self,
        assessment: FirstAssessment,
        source_audio: Optional[str] = None,
    ) -> AssessmentDocument:
        """Save a new FirstAssessment document to MongoDB.

        Args:
            assessment: FirstAssessment instance to persist.
            source_audio: Optional source audio file name.

        Returns:
            Persisted AssessmentDocument with generated ID and timestamps.
        """
        now = datetime.now(timezone.utc)
        doc_data: Dict[str, Any] = {
            "assessment": assessment.model_dump(),
            "created_at": now,
            "updated_at": now,
            "source_audio": source_audio,
        }

        try:
            result = await self.collection.insert_one(doc_data)
            doc_id = str(result.inserted_id)

            return AssessmentDocument(
                id=doc_id,
                assessment=assessment,
                created_at=now,
                updated_at=now,
                source_audio=source_audio,
            )
        except PyMongoError as exc:
            raise RepositoryException(f"Failed to insert assessment document: {str(exc)}") from exc

    async def get_by_id(self, assessment_id: str) -> Optional[AssessmentDocument]:
        """Retrieve an assessment by its MongoDB ObjectId.

        Args:
            assessment_id: String representation of MongoDB ObjectId.

        Returns:
            AssessmentDocument if found, None otherwise.
        """
        if not assessment_id or not ObjectId.is_valid(assessment_id):
            return None

        try:
            raw_doc = await self.collection.find_one({"_id": ObjectId(assessment_id)})
            if not raw_doc:
                return None

            return AssessmentDocument(
                id=str(raw_doc["_id"]),
                assessment=FirstAssessment.model_validate(raw_doc["assessment"]),
                created_at=raw_doc.get("created_at", datetime.now(timezone.utc)),
                updated_at=raw_doc.get("updated_at", datetime.now(timezone.utc)),
                source_audio=raw_doc.get("source_audio"),
            )
        except (InvalidId, PyMongoError) as exc:
            raise RepositoryException(f"Failed to retrieve assessment by ID: {str(exc)}") from exc

    async def list(
        self,
        start_date: Optional[Union[datetime, str]] = None,
        end_date: Optional[Union[datetime, str]] = None,
        skip: int = 0,
        limit: int = 20,
    ) -> Tuple[List[AssessmentDocument], int]:
        """List assessments with optional date-range filtering, sorting, and pagination.

        Args:
            start_date: Optional start datetime (inclusive).
            end_date: Optional end datetime (inclusive).
            skip: Number of documents to skip for pagination.
            limit: Maximum number of documents to return.

        Returns:
            Tuple of (List of AssessmentDocuments, total matching count).
        """
        query: Dict[str, Any] = {}
        date_filter: Dict[str, Any] = {}

        if start_date is not None:
            parsed_start = self._parse_datetime(start_date)
            date_filter["$gte"] = parsed_start

        if end_date is not None:
            parsed_end = self._parse_datetime(end_date)
            date_filter["$lte"] = parsed_end

        if date_filter:
            query["created_at"] = date_filter

        try:
            total_count = await self.collection.count_documents(query)

            cursor = (
                self.collection.find(query)
                .sort([("created_at", -1), ("_id", -1)])
                .skip(max(0, skip))
                .limit(max(1, limit))
            )

            documents: List[AssessmentDocument] = []
            async for raw_doc in cursor:
                doc = AssessmentDocument(
                    id=str(raw_doc["_id"]),
                    assessment=FirstAssessment.model_validate(raw_doc["assessment"]),
                    created_at=raw_doc.get("created_at", datetime.now(timezone.utc)),
                    updated_at=raw_doc.get("updated_at", datetime.now(timezone.utc)),
                    source_audio=raw_doc.get("source_audio"),
                )
                documents.append(doc)

            return documents, total_count
        except (ValueError, PyMongoError) as exc:
            raise RepositoryException(f"Failed to list assessments: {str(exc)}") from exc

    async def delete_by_id(self, assessment_id: str) -> bool:
        """Delete an assessment by ID (primarily used for test cleanup).

        Args:
            assessment_id: String representation of MongoDB ObjectId.

        Returns:
            True if document was deleted, False otherwise.
        """
        if not assessment_id or not ObjectId.is_valid(assessment_id):
            return False

        try:
            res = await self.collection.delete_one({"_id": ObjectId(assessment_id)})
            return res.deleted_count > 0
        except PyMongoError as exc:
            raise RepositoryException(f"Failed to delete assessment: {str(exc)}") from exc
