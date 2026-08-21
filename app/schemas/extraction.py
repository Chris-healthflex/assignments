from pydantic import BaseModel, ConfigDict, Field
from typing import Any, Optional

class ExtractionField(BaseModel):
    """A single extracted field with confidence and source quote."""
    model_config = ConfigDict(extra='forbid')
    value: Any = None
    is_mentioned: bool = False
    confidence: float = 0.0          # only meaningful when is_mentioned=True
    source_quote: str = ""

class ClinicalDetailsExtraction(BaseModel):
    model_config = ConfigDict(extra='forbid')
    clinicalHistory: ExtractionField = Field(default_factory=ExtractionField)
    chiefComplaint: ExtractionField = Field(default_factory=ExtractionField)
    duration: ExtractionField = Field(default_factory=ExtractionField)

class SubjectiveAssessmentExtraction(BaseModel):
    model_config = ConfigDict(extra='forbid')
    testName: ExtractionField = Field(default_factory=ExtractionField)
    conclusion: ExtractionField = Field(default_factory=ExtractionField)

class ObjectiveTestExtraction(BaseModel):
    model_config = ConfigDict(extra='forbid')
    testName: ExtractionField = Field(default_factory=ExtractionField)
    unitName: ExtractionField = Field(default_factory=ExtractionField)
    value: ExtractionField = Field(default_factory=ExtractionField)
    left: ExtractionField = Field(default_factory=ExtractionField)
    right: ExtractionField = Field(default_factory=ExtractionField)
    comments: ExtractionField = Field(default_factory=ExtractionField)

class ObjectiveAssessmentExtraction(BaseModel):
    model_config = ConfigDict(extra='forbid')
    tests: list[ObjectiveTestExtraction] = Field(default_factory=list)

class SubjectiveGoalExtraction(BaseModel):
    model_config = ConfigDict(extra='forbid')
    goalDetails: ExtractionField = Field(default_factory=ExtractionField)
    targetDate: ExtractionField = Field(default_factory=ExtractionField)

class ObjectiveGoalExtraction(BaseModel):
    model_config = ConfigDict(extra='forbid')
    goalName: ExtractionField = Field(default_factory=ExtractionField)
    goalCategory: ExtractionField = Field(default_factory=ExtractionField)
    unitName: ExtractionField = Field(default_factory=ExtractionField)
    value: ExtractionField = Field(default_factory=ExtractionField)
    targetDate: ExtractionField = Field(default_factory=ExtractionField)

class RecommendationExtraction(BaseModel):
    model_config = ConfigDict(extra='forbid')
    sessionType: ExtractionField = Field(default_factory=ExtractionField)
    sessionFrequency: ExtractionField = Field(default_factory=ExtractionField)

class PatientAdviceExtraction(BaseModel):
    model_config = ConfigDict(extra='forbid')
    adviceDetails: ExtractionField = Field(default_factory=ExtractionField)

class ExtractionResult(BaseModel):
    """Full internal extraction result for all sections."""
    model_config = ConfigDict(extra='forbid')
    clinicalDetails: ClinicalDetailsExtraction = Field(default_factory=ClinicalDetailsExtraction)
    subjectiveAssessments: list[SubjectiveAssessmentExtraction] = Field(default_factory=list)
    objectiveAssessment: ObjectiveAssessmentExtraction = Field(default_factory=ObjectiveAssessmentExtraction)
    subjectiveGoals: list[SubjectiveGoalExtraction] = Field(default_factory=list)
    objectiveGoals: list[ObjectiveGoalExtraction] = Field(default_factory=list)
    recommendation: list[RecommendationExtraction] = Field(default_factory=list)
    patientAdvice: PatientAdviceExtraction = Field(default_factory=PatientAdviceExtraction)