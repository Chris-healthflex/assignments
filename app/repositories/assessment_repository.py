from datetime import datetime, timezone
from typing import Any
import uuid
from bson import ObjectId
from bson.errors import InvalidId
from pymongo.collection import Collection
from pymongo.errors import PyMongoError

from app.core.config import get_settings
from app.core.logging import logger
from app.core.errors import DatabaseError, AssessmentNotFoundError
from app.database.mongodb import get_db
from app.schemas.first_assessment import FirstAssessment
from app.models.assessment import AssessmentDocument


class AssessmentRepository:
    """Repository layer managing persistence and querying of FirstAssessment records."""

    def __init__(self, collection: Collection[dict[str, Any]] | None = None) -> None:
        self._collection = collection

    @property
    def collection(self) -> Collection[dict[str, Any]]:
        if self._collection is not None:
            return self._collection
        db = get_db()
        settings = get_settings()
        return db[settings.MONGODB_COLLECTION]

    def save_assessment(self, assessment: FirstAssessment) -> AssessmentDocument:
        """Persists a FirstAssessment document into MongoDB."""
        try:
            doc_id = str(uuid.uuid4())
            now = datetime.now(timezone.utc)

            doc_data = {
                "_id": doc_id,
                "created_at": now.isoformat(),
                "assessment": assessment.model_dump(),
            }

            self.collection.insert_one(doc_data)
            logger.info("Assessment saved with ID: %s", doc_id)

            return AssessmentDocument(
                id=doc_id,
                created_at=now,
                assessment=assessment
            )
        except PyMongoError as exc:
            logger.error("Failed to save assessment to MongoDB: %s", exc)
            raise DatabaseError("Database failure while saving assessment", details=str(exc))

    def get_assessment(self, assessment_id: str) -> AssessmentDocument:
        """Retrieves a single assessment by its ID."""
        try:
            # Query by string id or ObjectId if valid
            query: dict[str, Any] = {"_id": assessment_id}
            doc = self.collection.find_one(query)

            if not doc and ObjectId.is_valid(assessment_id):
                try:
                    doc = self.collection.find_one({"_id": ObjectId(assessment_id)})
                except InvalidId:
                    pass

            if not doc:
                logger.warning("Assessment with ID '%s' not found", assessment_id)
                raise AssessmentNotFoundError(assessment_id=assessment_id)

            created_at_val = doc.get("created_at")
            if isinstance(created_at_val, str):
                created_at = datetime.fromisoformat(created_at_val)
            elif isinstance(created_at_val, datetime):
                created_at = created_at_val
            else:
                created_at = datetime.now(timezone.utc)

            raw_assessment = doc.get("assessment", {})
            first_assessment = FirstAssessment.model_validate(raw_assessment)

            return AssessmentDocument(
                id=str(doc.get("_id")),
                created_at=created_at,
                assessment=first_assessment
            )
        except AssessmentNotFoundError:
            raise
        except PyMongoError as exc:
            logger.error("Failed to retrieve assessment '%s': %s", assessment_id, exc)
            raise DatabaseError("Database failure while retrieving assessment", details=str(exc))

    def list_assessments(self, date_filter: str | None = None) -> list[AssessmentDocument]:
        """Lists stored assessments, optionally filtered by creation date (YYYY-MM-DD)."""
        try:
            query: dict[str, Any] = {}
            if date_filter:
                # Filter by date prefix in ISO string or start/end of day
                query["created_at"] = {"$regex": f"^{date_filter}"}

            cursor = self.collection.find(query).sort("created_at", -1)
            results: list[AssessmentDocument] = []

            for doc in cursor:
                created_at_val = doc.get("created_at")
                if isinstance(created_at_val, str):
                    created_at = datetime.fromisoformat(created_at_val)
                elif isinstance(created_at_val, datetime):
                    created_at = created_at_val
                else:
                    created_at = datetime.now(timezone.utc)

                raw_assessment = doc.get("assessment", {})
                first_assessment = FirstAssessment.model_validate(raw_assessment)
                results.append(
                    AssessmentDocument(
                        id=str(doc.get("_id")),
                        created_at=created_at,
                        assessment=first_assessment
                    )
                )

            logger.info("Retrieved %d assessment records (filter=%s)", len(results), date_filter)
            return results
        except PyMongoError as exc:
            logger.error("Failed to list assessments from MongoDB: %s", exc)
            raise DatabaseError("Database failure while listing assessments", details=str(exc))


def get_assessment_repository() -> AssessmentRepository:
    """Dependency provider for repository instance."""
    return AssessmentRepository()
