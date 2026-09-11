from pydantic import BaseModel
from typing import List
from app.models.assessment import FirstAssessment

class ParseResponse(BaseModel):
    id: str
    assessment: FirstAssessment

class ValidationErrorDetail(BaseModel):
    field: str
    reason: str
    confidence: float

class LowConfidenceResponse(BaseModel):
    detail: List[ValidationErrorDetail]
