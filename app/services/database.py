import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from bson import ObjectId
from bson.errors import InvalidId

from app.config import settings
from app.models.schema import FirstAssessment

logger = logging.getLogger(__name__)


class Database:
    def __init__(self):
        self.client = None
        self.db = None
        self.collection = None
        self.is_mock = False
        self._mock_storage: Dict[str, Dict[str, Any]] = {}

    def connect(self):
        try:
            from pymongo import MongoClient
            # Attempt short timeout ping to live MongoDB
            client = MongoClient(settings.MONGODB_URI, serverSelectionTimeoutMS=1500)
            client.admin.command("ping")
            self.client = client
            self.db = self.client[settings.MONGODB_DB_NAME]
            self.collection = self.db[settings.MONGODB_COLLECTION]
            self.is_mock = False
            logger.info(f"Connected to live MongoDB at {settings.MONGODB_URI}")
        except Exception as e:
            logger.warning(f"Live MongoDB connection failed ({e}). Initializing in-memory mock storage.")
            self.is_mock = True

    def get_collection(self):
        if self.client is None and not self.is_mock:
            self.connect()
        return self.collection

    async def save_assessment(self, assessment: FirstAssessment) -> Dict[str, Any]:
        """
        Saves a parsed FirstAssessment document into MongoDB.
        """
        if self.client is None and not self.is_mock:
            self.connect()

        now_iso = datetime.now(timezone.utc).isoformat()
        doc = {
            "assessment": assessment.model_dump(),
            "created_at": now_iso,
            "updated_at": now_iso,
        }

        if self.is_mock:
            doc_id = str(ObjectId())
            doc["_id"] = doc_id
            self._mock_storage[doc_id] = doc
            return {
                "id": doc_id,
                "assessment": assessment,
                "created_at": now_iso,
            }

        try:
            result = self.collection.insert_one(doc)
            return {
                "id": str(result.inserted_id),
                "assessment": assessment,
                "created_at": now_iso,
            }
        except Exception as e:
            logger.error(f"Failed to insert assessment to MongoDB: {e}")
            raise

    async def get_assessment_by_id(self, assessment_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves a single assessment by its MongoDB ObjectID string.
        """
        if self.client is None and not self.is_mock:
            self.connect()

        if self.is_mock:
            doc = self._mock_storage.get(assessment_id)
            if not doc:
                return None
            return {
                "id": str(doc["_id"]),
                "assessment": doc["assessment"],
                "created_at": doc.get("created_at", ""),
                "updated_at": doc.get("updated_at", ""),
            }

        try:
            oid = ObjectId(assessment_id)
        except InvalidId:
            return None

        doc = self.collection.find_one({"_id": oid})
        if not doc:
            return None

        return {
            "id": str(doc["_id"]),
            "assessment": doc["assessment"],
            "created_at": doc.get("created_at", ""),
            "updated_at": doc.get("updated_at", ""),
        }

    async def list_assessments(
        self,
        date_str: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        skip: int = 0,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Retrieves assessments with optional date filtering.
        """
        if self.client is None and not self.is_mock:
            self.connect()

        if self.is_mock:
            results = list(self._mock_storage.values())
            # Date filter
            if date_str:
                results = [r for r in results if r.get("created_at", "").startswith(date_str)]
            if start_date:
                results = [r for r in results if r.get("created_at", "") >= start_date]
            if end_date:
                results = [r for r in results if r.get("created_at", "") <= end_date]
            
            # Pagination
            paged = results[skip : skip + limit]
            return [
                {
                    "id": str(r["_id"]),
                    "assessment": r["assessment"],
                    "created_at": r.get("created_at", ""),
                }
                for r in paged
            ]

        query: Dict[str, Any] = {}
        if date_str:
            query["created_at"] = {"$regex": f"^{date_str}"}
        elif start_date or end_date:
            date_filter: Dict[str, Any] = {}
            if start_date:
                date_filter["$gte"] = start_date
            if end_date:
                date_filter["$lte"] = end_date
            query["created_at"] = date_filter

        cursor = self.collection.find(query).skip(skip).limit(limit)
        items = []
        for doc in cursor:
            items.append({
                "id": str(doc["_id"]),
                "assessment": doc["assessment"],
                "created_at": doc.get("created_at", ""),
            })
        return items


# Singleton database instance
db = Database()
