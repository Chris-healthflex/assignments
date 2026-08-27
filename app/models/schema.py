from typing import List, Optional, Dict, Any
from pydantic import BaseModel, ConfigDict, Field, field_validator


class ClinicalDetails(BaseModel):
    model_config = ConfigDict(extra="forbid")

    clinicalHistory: str = Field(default="", description="Patient clinical history")
    chiefComplaint: str = Field(default="", description="Primary complaint of patient")
    duration: str = Field(default="", description="Duration of symptoms")

    @field_validator("*", mode="before")
    @classmethod
    def ensure_string(cls, v: Any) -> str:
        if v is None:
            return ""
        return str(v)


class SubjectiveAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    testName: str = Field(default="", description="Subjective assessment or test name")
    conclusion: str = Field(default="", description="Clinical conclusion or finding")

    @field_validator("*", mode="before")
    @classmethod
    def ensure_string(cls, v: Any) -> str:
        if v is None:
            return ""
        return str(v)


class ObjectiveTest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    testName: str = Field(default="", description="Objective physical or clinical test name")
    unitName: str = Field(default="", description="Unit of measurement if applicable")
    value: str = Field(default="", description="Measurement value or range")
    left: str = Field(default="", description="Left side measurement/finding")
    right: str = Field(default="", description="Right side measurement/finding")
    comments: str = Field(default="", description="Clinician observations or comments")

    @field_validator("*", mode="before")
    @classmethod
    def ensure_string(cls, v: Any) -> str:
        if v is None:
            return ""
        return str(v)


class ObjectiveAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tests: List[ObjectiveTest] = Field(default_factory=list, description="Array of objective tests")


class SubjectiveGoal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    goalDetails: str = Field(default="", description="Subjective goal details")
    targetDate: str = Field(default="", description="Target achievement timeframe or date")

    @field_validator("*", mode="before")
    @classmethod
    def ensure_string(cls, v: Any) -> str:
        if v is None:
            return ""
        return str(v)


class ObjectiveGoal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    goalName: str = Field(default="", description="Name of measurable goal")
    goalCategory: str = Field(default="", description="Category of objective goal")
    unitName: str = Field(default="", description="Measurement unit")
    value: str = Field(default="", description="Target measurable value")
    targetDate: str = Field(default="", description="Target timeline or date")

    @field_validator("*", mode="before")
    @classmethod
    def ensure_string(cls, v: Any) -> str:
        if v is None:
            return ""
        return str(v)


class Recommendation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sessionType: str = Field(default="", description="Type of clinical session or therapy")
    sessionFrequency: str = Field(default="", description="Frequency of sessions")

    @field_validator("*", mode="before")
    @classmethod
    def ensure_string(cls, v: Any) -> str:
        if v is None:
            return ""
        return str(v)


class PatientAdvice(BaseModel):
    model_config = ConfigDict(extra="forbid")

    adviceDetails: str = Field(default="", description="Instructions or advice for patient")

    @field_validator("*", mode="before")
    @classmethod
    def ensure_string(cls, v: Any) -> str:
        if v is None:
            return ""
        return str(v)


class FirstAssessment(BaseModel):
    """
    Exact production schema consumed by frontend.
    Contains 7 mandatory sections with strict types (strings and arrays only, no nulls, no extra keys).
    """
    model_config = ConfigDict(extra="forbid")

    clinicalDetails: ClinicalDetails = Field(default_factory=ClinicalDetails)
    subjectiveAssessments: List[SubjectiveAssessment] = Field(default_factory=list)
    objectiveAssessment: ObjectiveAssessment = Field(default_factory=ObjectiveAssessment)
    subjectiveGoals: List[SubjectiveGoal] = Field(default_factory=list)
    objectiveGoals: List[ObjectiveGoal] = Field(default_factory=list)
    recommendation: List[Recommendation] = Field(default_factory=list)
    patientAdvice: PatientAdvice = Field(default_factory=PatientAdvice)


class ExtractionConfidence(BaseModel):
    overall_score: float = Field(ge=0.0, le=1.0)
    section_scores: Dict[str, float] = Field(default_factory=dict)
    flagged_fields: List[str] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)


class ExtractionResponse(BaseModel):
    assessment: FirstAssessment
    confidence: ExtractionConfidence
    transcription: str


class AssessmentSaveResponse(BaseModel):
    id: str
    assessment: FirstAssessment
    created_at: str
    message: str = "Assessment saved successfully"


class AssessmentListResponse(BaseModel):
    total: int
    items: List[Dict[str, Any]]
