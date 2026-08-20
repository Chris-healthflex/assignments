from __future__ import annotations
from datetime import datetime
from typing import List, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.assessment import FirstAssessment


class FieldConfidence(BaseModel):

    field: str
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: str = ""


class ExtractionMeta(BaseModel):

    transcript: str = ""
    transcriptLanguage: str = ""
    audioDurationSeconds: float = 0.0
    sourceFilename: str = ""
    overallConfidence: float = 0.0
    fieldConfidence: List[FieldConfidence] = Field(default_factory=list)
    unextractedFields: List[str] = Field(default_factory=list)
    groundingWarnings: List[str] = Field(default_factory=list)
    extractionErrors: List[str] = Field(default_factory=list)
    llmProvider: str = ""
    llmModel: str = ""
    attempts: int = 0


class ParseResponse(BaseModel):

    assessment: FirstAssessment
    meta: ExtractionMeta


class SaveAssessmentRequest(BaseModel):

    model_config = ConfigDict(extra="forbid")

    assessment: FirstAssessment
    meta: ExtractionMeta | None = None


class StoredAssessment(BaseModel):

    id: str
    createdAt: datetime
    assessment: FirstAssessment
    meta: ExtractionMeta | None = None


class AssessmentListResponse(BaseModel):
    total: int
    count: int
    items: List[StoredAssessment]


class LowConfidenceDetail(BaseModel):

    message: str
    overallConfidence: float
    threshold: float
    fields: List[FieldConfidence] = Field(default_factory=list)
    unextractedFields: List[str] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    mongo: str
    whisperBackend: str
    llmProvider: str
