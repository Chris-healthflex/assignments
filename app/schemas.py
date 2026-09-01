from typing import List
from pydantic import BaseModel, Field


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
    numberOfSessions: str = ""


class PatientAdvice(BaseModel):
    adviceDetails: str = ""


class FirstAssessment(BaseModel):
    clinicalDetails: ClinicalDetails
    subjectiveAssessments: List[SubjectiveAssessment]
    objectiveAssessment: ObjectiveAssessment
    subjectiveGoals: List[SubjectiveGoal]
    objectiveGoals: List[ObjectiveGoal]
    recommendation: List[Recommendation]
    patientAdvice: PatientAdvice