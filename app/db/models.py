from pydantic import BaseModel, ConfigDict, Field
from typing import Any, Optional
from datetime import datetime
from bson import ObjectId

class AssessmentDocument(BaseModel):
    model_config = ConfigDict(extra='forbid', arbitrary_types_allowed=True)
    id: Optional[str] = Field(default=None, alias="_id")
    assessment: dict[str, Any]
    transcript: Optional[str] = None
    pipeline_version: str = "v2"
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    @classmethod
    def from_mongo(cls, doc: dict) -> "AssessmentDocument":
        if "_id" in doc:
            doc["_id"] = str(doc["_id"])
        return cls(**doc)