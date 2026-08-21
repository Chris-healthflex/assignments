from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from app.models.first_assessment import FirstAssessment


class AssessmentRecord(FirstAssessment):
    model_config = ConfigDict(extra="forbid")
    id: str
    created_at: datetime


class ConfidenceIssue(BaseModel):
    field: str
    confidence: float = Field(ge=0, le=1)
    message: str


class ConfidenceError(BaseModel):
    detail: str
    fields: list[ConfidenceIssue]
