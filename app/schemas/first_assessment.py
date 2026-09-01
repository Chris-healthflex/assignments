from pydantic import BaseModel, Field
from typing import List


class ClinicalDetails(BaseModel):
    clinicalHistory: str = ""
    chiefComplaint: str = ""
    duration: str = ""


class SubjectiveAssessment(BaseModel):
    testName: str = ""
    conclusion: str = ""


class ObjectiveTest(BaseModel):
    testName: str = ""
    unitName: str = ""
    value: str = ""
    left: str = ""
    right: str = ""
    comments: str = ""


class ObjectiveAssessment(BaseModel):
    tests: List[ObjectiveTest] = Field(default_factory=list)


class SubjectiveGoal(BaseModel):
    goalDetails: str = ""
    targetDate: str = ""


class ObjectiveGoal(BaseModel):
    goalName: str = ""
    goalCategory: str = ""
    unitName: str = ""
    value: str = ""
    targetDate: str = ""


class Recommendation(BaseModel):
    sessionType: str = ""
    sessionFrequency: str = ""


class PatientAdvice(BaseModel):
    adviceDetails: str = ""


class FirstAssessment(BaseModel):
    clinicalDetails: ClinicalDetails
    subjectiveAssessments: List[SubjectiveAssessment] = Field(default_factory=list)
    objectiveAssessment: ObjectiveAssessment
    subjectiveGoals: List[SubjectiveGoal] = Field(default_factory=list)
    objectiveGoals: List[ObjectiveGoal] = Field(default_factory=list)
    recommendation: List[Recommendation] = Field(default_factory=list)
    patientAdvice: PatientAdvice
    flaggedFields: List[str] = Field(
        default_factory=list,
        description="Fields that could not be confidently extracted from the transcript and were left empty.",
    )