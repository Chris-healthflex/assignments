from datetime import datetime, timezone
from typing import Any, Optional

from bson import ObjectId
from pymongo.collection import Collection

from app.database import get_database
from app.models.assessment import FirstAssessment


class AssessmentRepository:
    def __init__(self) -> None:
        self.db = get_database()
        self.collection: Collection = self.db["assessments"]

        self.collection.create_index(
            [("created_at", -1)]
        )

    def create(self, assessment: FirstAssessment) -> str:
        document = {
            "assessment": assessment.model_dump(mode="json"),
            "created_at": datetime.now(timezone.utc),
        }

        result = self.collection.insert_one(document)

        return str(result.inserted_id)

    def get_by_id(self, assessment_id: str) -> Optional[dict[str, Any]]:
        if not ObjectId.is_valid(assessment_id):
            return None

        document = self.collection.find_one(
            {"_id": ObjectId(assessment_id)}
        )

        if document is None:
            return None

        return {
            "id": str(document["_id"]),
            "assessment": document["assessment"],
            "created_at": document["created_at"],
        }

    def list(
        self,
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None,
    ) -> list[dict[str, Any]]:
        query: dict[str, Any] = {}

        if from_date or to_date:
            created_at_query: dict[str, Any] = {}

            if from_date:
                created_at_query["$gte"] = from_date

            if to_date:
                created_at_query["$lte"] = to_date

            query["created_at"] = created_at_query

        documents = self.collection.find(query).sort(
            "created_at",
            -1,
        )

        results: list[dict[str, Any]] = []

        for document in documents:
            results.append(
                {
                    "id": str(document["_id"]),
                    "assessment": document["assessment"],
                    "created_at": document["created_at"],
                }
            )

        return results
