"""Request and response models for the API.

The central design point: ``FirstAssessment`` forbids extra keys, so the S5
confidence flags cannot live inside it. They travel as siblings in an
envelope, leaving ``assessment`` exactly seven keys wide and directly usable
by the frontend.

Callers that want nothing but the bare assessment can pass ``?envelope=false``
on the parse endpoint, which satisfies the strictest reading of the brief
without hiding the flag report from everyone else.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from app.db.models import AssessmentMetadata
from app.extraction.confidence import FieldFlag
from app.schemas.first_assessment import FirstAssessment


class TranscriptInfo(BaseModel):
    """What Whisper heard. Returned so a clinician can audit the extraction."""

    text: str = ""
    language: str = ""
    durationSeconds: float = 0.0
    segments: int = 0
    model: str = ""
    backend: str = ""


class ConfidenceInfo(BaseModel):
    overall: float = 0.0
    threshold: float = 0.0
    meetsThreshold: bool = False
    sectionScores: dict[str, float] = Field(default_factory=dict)
    rejectedCount: int = 0


class ParseResponse(BaseModel):
    """The parse result: the assessment plus everything needed to audit it."""

    assessment: FirstAssessment
    transcript: TranscriptInfo
    confidence: ConfidenceInfo
    flaggedFields: list[FieldFlag] = Field(default_factory=list)
    timings: dict[str, float] = Field(default_factory=dict)

    model_config = {
        "json_schema_extra": {
            "example": {
                "assessment": {
                    "clinicalDetails": {
                        "clinicalHistory": "Road traffic accident eight months ago resulting in a left tibial condyle fracture and an avulsion ACL tear. Open reduction and internal fixation performed by Dr. Hemant Kalyan.",
                        "chiefComplaint": "left knee pain, difficulty performing functional activities and difficulty walking",
                        "duration": "eight months",
                    },
                    "subjectiveAssessments": [
                        {
                            "testName": "Pain",
                            "conclusion": "moderate pain with mild irritability during prolonged walking and standing, relieved with rest",
                        }
                    ],
                    "objectiveAssessment": {
                        "tests": [
                            {
                                "testName": "Knee flexion",
                                "unitName": "degrees",
                                "value": "",
                                "left": "124",
                                "right": "130",
                                "comments": "",
                            }
                        ]
                    },
                    "subjectiveGoals": [],
                    "objectiveGoals": [
                        {
                            "goalName": "Restoring the extension",
                            "goalCategory": "",
                            "unitName": "",
                            "value": "",
                            "targetDate": "",
                        }
                    ],
                    "recommendation": [
                        {
                            "sessionType": "Physiotherapy",
                            "sessionFrequency": "once weekly for four sessions",
                        }
                    ],
                    "patientAdvice": {"adviceDetails": ""},
                },
                "transcript": {
                    "text": "The patient presented with left knee pain...",
                    "language": "en",
                    "durationSeconds": 105.55,
                    "segments": 24,
                    "model": "small",
                    "backend": "faster-whisper",
                },
                "confidence": {
                    "overall": 0.9,
                    "threshold": 0.55,
                    "meetsThreshold": True,
                    "sectionScores": {"clinicalDetails": 1.0, "objectiveAssessment": 1.0},
                    "rejectedCount": 0,
                },
                "flaggedFields": [
                    {
                        "path": "objectiveGoals[0].targetDate",
                        "reason": "not_stated",
                        "detail": "",
                    },
                    {
                        "path": "patientAdvice.adviceDetails",
                        "reason": "not_stated",
                        "detail": "",
                    },
                ],
                "timings": {"transcribe": 24.3, "extract": 129.2, "total": 153.5},
            }
        }
    }


class SaveAssessmentRequest(BaseModel):
    """Body for persisting a parsed result."""

    assessment: FirstAssessment
    metadata: AssessmentMetadata = Field(default_factory=AssessmentMetadata)


class SavedAssessmentResponse(BaseModel):
    id: str
    createdAt: datetime
    assessment: FirstAssessment
    metadata: AssessmentMetadata


class AssessmentListResponse(BaseModel):
    total: int
    count: int
    limit: int
    skip: int
    items: list[SavedAssessmentResponse] = Field(default_factory=list)


class LowConfidenceDetail(BaseModel):
    """Body of the 422 returned when extraction confidence is too low.

    The partial assessment and the transcript are included deliberately: a
    clinician looking at a rejected parse needs to see what was heard and what
    was extracted in order to judge whether to retry or complete it by hand.
    """

    message: str
    confidence: float
    threshold: float
    fields: list[FieldFlag] = Field(default_factory=list)
    transcript: str = ""
    assessment: dict[str, Any] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    status: str
    mongodb: bool
    llmProvider: str
    llmModel: str
    llmReachable: bool
    whisperBackend: str
    whisperModel: str
    whisperLoaded: bool


class ErrorResponse(BaseModel):
    detail: str
