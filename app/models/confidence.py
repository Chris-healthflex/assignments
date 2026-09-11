from pydantic import BaseModel, Field, ConfigDict

from app.models.assessment import FirstAssessment


class ConfidenceReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    clinicalDetails: float = Field(ge=0.0, le=1.0)
    subjectiveAssessments: float = Field(ge=0.0, le=1.0)
    objectiveAssessment: float = Field(ge=0.0, le=1.0)
    subjectiveGoals: float = Field(ge=0.0, le=1.0)
    objectiveGoals: float = Field(ge=0.0, le=1.0)
    recommendation: float = Field(ge=0.0, le=1.0)
    patientAdvice: float = Field(ge=0.0, le=1.0)


class ExtractionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    assessment: FirstAssessment
    confidence: ConfidenceReport