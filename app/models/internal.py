"""Internal models used by the agent and API."""
from typing import Dict, List, Optional
from datetime import datetime
from pydantic import BaseModel, Field
from app.models.assessment import FirstAssessment

class ExtractionEnvelope(BaseModel):
    assessment: FirstAssessment
    field_confidence: Dict[str, float] = Field(default_factory=dict)

class LowConfidenceField(BaseModel):
    field: str
    confidence: float
    threshold: float

class PipelineResult(BaseModel):
    assessment: Optional[FirstAssessment] = None
    field_confidence: Dict[str, float] = Field(default_factory=dict)
    low_confidence_fields: List[LowConfidenceField] = Field(default_factory=list)
    transcript: str
    error: Optional[str] = None

    @property
    def passed(self) -> bool:
        return not self.error and not self.low_confidence_fields

class StoredAssessment(BaseModel):
    id: str
    assessment: FirstAssessment
    created_at: datetime
    source_transcript: Optional[str] = None

class LowConfidenceErrorDetail(BaseModel):
    message: str
    threshold: float
    fields: List[LowConfidenceField]

class LowConfidenceErrorResponse(BaseModel):
    detail: LowConfidenceErrorDetail

