import os
from datetime import datetime, timezone

from dotenv import load_dotenv
from pymongo import MongoClient
from bson import ObjectId

from app.models.assessment import FirstAssessment


load_dotenv()


class MongoDB:
    def __init__(self):
        uri = os.getenv("MONGODB_URI")

        if not uri:
            raise ValueError(
                "MONGODB_URI was not found. Check your .env file."
            )

        database_name = os.getenv(
            "MONGODB_DATABASE",
            "clinical_assessment_db",
        )

        collection_name = os.getenv(
            "MONGODB_COLLECTION",
            "assessments",
        )

        self.client = MongoClient(
            uri,
            serverSelectionTimeoutMS=5000,
        )

        self.database = self.client[database_name]
        self.collection = self.database[collection_name]

    def save_assessment(self, assessment: FirstAssessment) -> str:
        data = assessment.model_dump()
        data["createdAt"] = datetime.now(timezone.utc)
        result = self.collection.insert_one(data)
        return str(result.inserted_id)
        
    def get_assessment(self, assessment_id: str) -> dict | None:
        result = self.collection.find_one({"_id": ObjectId(assessment_id)})
        if result:
            result["_id"] = str(result["_id"])
        return result