import uuid
from datetime import datetime, timezone
from typing import List, Optional, Tuple, Dict, Any
from app.db.mongo import get_database
from app.config import settings
from app.models.assessment import FirstAssessment
from app.models.internal import AssessmentRecord, ConfidenceReport


class AssessmentRepository:
    def __init__(self):
        self.db = get_database()
        self.collection = self.db[settings.COLLECTION_NAME]

    async def create(
        self,
        assessment: FirstAssessment,
        confidence: Optional[ConfidenceReport] = None,
        assessment_id: Optional[str] = None,
    ) -> AssessmentRecord:
        """Inserts an assessment document into MongoDB and returns an AssessmentRecord."""
        record_id = assessment_id or str(uuid.uuid4())
        created_at = datetime.now(timezone.utc)

        doc: Dict[str, Any] = {
            "_id": record_id,
            "createdAt": created_at,
            "assessment": assessment.model_dump(),
            "confidence": confidence.model_dump() if confidence else None,
        }

        await self.collection.insert_one(doc)

        return AssessmentRecord(
            id=record_id,
            createdAt=created_at,
            assessment=assessment,
            confidence=confidence,
        )

    async def get_by_id(self, assessment_id: str) -> Optional[AssessmentRecord]:
        """Fetches an assessment document by its ID."""
        doc = await self.collection.find_one({"_id": assessment_id})
        if not doc:
            return None

        return AssessmentRecord(
            id=str(doc["_id"]),
            createdAt=doc["createdAt"],
            assessment=FirstAssessment.model_validate(doc["assessment"]),
            confidence=ConfidenceReport.model_validate(doc["confidence"]) if doc.get("confidence") else None,
        )

    async def list_all(
        self,
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None,
        skip: int = 0,
        limit: int = 20,
    ) -> Tuple[List[AssessmentRecord], int]:
        """Queries assessments with optional date filtering, pagination, and total count."""
        query: Dict[str, Any] = {}

        if from_date or to_date:
            date_filter: Dict[str, Any] = {}
            if from_date:
                date_filter["$gte"] = from_date
            if to_date:
                date_filter["$lte"] = to_date
            query["createdAt"] = date_filter

        total = await self.collection.count_documents(query)
        cursor = self.collection.find(query).sort("createdAt", -1).skip(skip).limit(limit)
        docs = await cursor.to_list(length=limit)

        records: List[AssessmentRecord] = []
        for doc in docs:
            records.append(
                AssessmentRecord(
                    id=str(doc["_id"]),
                    createdAt=doc["createdAt"],
                    assessment=FirstAssessment.model_validate(doc["assessment"]),
                    confidence=ConfidenceReport.model_validate(doc["confidence"]) if doc.get("confidence") else None,
                )
            )

        return records, total
