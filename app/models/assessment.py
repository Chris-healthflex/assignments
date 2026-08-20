from datetime import datetime, timezone
from typing import Any
from pydantic import BaseModel, Field


class AssessmentDocument(BaseModel):
    """MongoDB storage document representation."""
    id: str = Field(..., validation_alias="_id", serialization_alias="id")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    assessment: Any  # FirstAssessment

    model_config = {
        "populate_by_name": True,
    }


class SaveAssessmentResponse(BaseModel):
    """Response returned upon saving an assessment."""
    id: str = Field(..., description="Unique assessment document ID")
    message: str = Field(default="Assessment saved successfully")
    created_at: str = Field(..., description="ISO 8601 creation timestamp")


class AssessmentSummaryItem(BaseModel):
    """Summary item returned in assessment list."""
    id: str = Field(..., description="Unique assessment document ID")
    created_at: str = Field(..., description="ISO 8601 creation timestamp")
    chiefComplaint: str = Field(default="", description="Chief complaint summary")
    assessment: Any = Field(..., description="Stored FirstAssessment object")


class AssessmentListResponse(BaseModel):
    """List response containing stored assessments."""
    total: int
    assessments: list[AssessmentSummaryItem]
