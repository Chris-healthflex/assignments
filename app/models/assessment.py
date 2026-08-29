"""Output schema for the FirstAssessment (schema/v1)."""
from typing import List, Optional
from pydantic import BaseModel, Field

SECTION_ALIASES = {
    "clinical_details": "clinicalDetails",
    "subjective_assessments": "subjectiveAssessments",
    "objective_assessment": "objectiveAssessment",
    "subjective_goals": "subjectiveGoals",
    "objective_goals": "objectiveGoals",
    "recommendation": "recommendation",
    "patient_advice": "patientAdvice"
}

class ClinicalDetails(BaseModel):
    clinicalHistory: Optional[str] = None
    chiefComplaint: Optional[str] = None
    duration: Optional[str] = None

class SubjectiveAssessmentItem(BaseModel):
    testName: Optional[str] = None
    conclusion: Optional[str] = None

class ObjectiveTestItem(BaseModel):
    testName: Optional[str] = None
    unitName: Optional[str] = None
    value: Optional[str] = None
    left: Optional[str] = None
    right: Optional[str] = None
    comments: Optional[str] = None

class ObjectiveAssessment(BaseModel):
    tests: List[ObjectiveTestItem] = Field(default_factory=list)

class SubjectiveGoalItem(BaseModel):
    goalDetails: Optional[str] = None
    targetDate: Optional[str] = None

class ObjectiveGoalItem(BaseModel):
    goalName: Optional[str] = None
    goalCategory: Optional[str] = None
    unitName: Optional[str] = None
    value: Optional[str] = None
    targetDate: Optional[str] = None

class RecommendationItem(BaseModel):
    sessionType: Optional[str] = None
    sessionFrequency: Optional[str] = None

class PatientAdvice(BaseModel):
    adviceDetails: Optional[str] = None

class FirstAssessment(BaseModel):
    clinicalDetails: Optional[ClinicalDetails] = None
    subjectiveAssessments: List[SubjectiveAssessmentItem] = Field(default_factory=list)
    objectiveAssessment: Optional[ObjectiveAssessment] = None
    subjectiveGoals: List[SubjectiveGoalItem] = Field(default_factory=list)
    objectiveGoals: List[ObjectiveGoalItem] = Field(default_factory=list)
    recommendation: List[RecommendationItem] = Field(default_factory=list)
    patientAdvice: Optional[PatientAdvice] = None
