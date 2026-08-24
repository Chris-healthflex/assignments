"""Request/response models for the HTTP layer.

These wrap the core FirstAssessment contract with the pipeline metadata
(transcript, confidence, flagged fields, timings) and the DB envelope.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.assessment import FirstAssessment


class TranscriptMeta(BaseModel):
    text: str = ""
    language: str = ""
    durationSeconds: float = 0.0
    segments: int = 0
    model: str = ""
    backend: str = ""


class FlaggedField(BaseModel):
    path: str
    reason: str          # not_stated | ungrounded | low_confidence
    detail: str = ""


class ConfidenceReport(BaseModel):
    overall: float = 0.0
    threshold: float = 0.0
    meetsThreshold: bool = False
    sectionScores: Dict[str, float] = Field(default_factory=dict)
    rejectedCount: int = 0


class PipelineResult(BaseModel):
    """Full envelope returned by the audio->JSON pipeline. Mirrors sample_output.json."""

    assessment: FirstAssessment
    transcript: TranscriptMeta
    confidence: ConfidenceReport
    flaggedFields: List[FlaggedField] = Field(default_factory=list)
    timings: Dict[str, float] = Field(default_factory=dict)


class AssessmentCreate(BaseModel):
    """Body for POST /assessments — accepts a full pipeline result to persist."""

    model_config = ConfigDict(extra="allow")

    assessment: FirstAssessment
    transcript: Optional[TranscriptMeta] = None
    confidence: Optional[ConfidenceReport] = None
    flaggedFields: List[FlaggedField] = Field(default_factory=list)
    timings: Dict[str, float] = Field(default_factory=dict)


class StoredAssessment(BaseModel):
    """What GET endpoints return: the record plus its Mongo id and timestamp."""

    id: str
    createdAt: datetime
    assessment: FirstAssessment
    transcript: Optional[TranscriptMeta] = None
    confidence: Optional[ConfidenceReport] = None
    flaggedFields: List[FlaggedField] = Field(default_factory=list)
    timings: Dict[str, float] = Field(default_factory=dict)


class AssessmentList(BaseModel):
    count: int
    items: List[StoredAssessment]
