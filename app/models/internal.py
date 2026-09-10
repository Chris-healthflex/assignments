from __future__ import annotations
from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field

from app.models.assessment import FirstAssessment


class FieldEvidence(BaseModel):
    field: str  # dot-path, e.g. "objectiveAssessment.tests[0].value"
    confidence: float  # 0.0-1.0, post-fusion
    llm_confidence: float  # raw self-reported score, pre-fusion
    grounded: bool  # did deterministic matching find support?
    evidence_span: Optional[str] = None  # the transcript substring it was matched against, if any
    reason: str = ""  # short human-readable explanation


class ConfidenceReport(BaseModel):
    overall: float
    threshold: float
    flags: List[FieldEvidence] = Field(default_factory=list)  # only fields below threshold or ungrounded numerics
    passed: bool


class AssessmentRecord(BaseModel):
    """Returned by GET /assessments/{id} and POST /assessments: a wrapper, not the raw contract."""
    id: str
    createdAt: datetime
    assessment: FirstAssessment
    confidence: Optional[ConfidenceReport] = None


class AssessmentListResponse(BaseModel):
    """Paginated response for GET /assessments."""
    items: List[AssessmentRecord]
    total: int
    limit: int
    skip: int
