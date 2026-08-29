from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class StrictModel(BaseModel):
    """Base class: forbids unknown fields on every nested model."""

    model_config = ConfigDict(extra="forbid")


class ClinicalDetails(StrictModel):
    clinicalHistory: str
    chiefComplaint: str
    duration: str


class SubjectiveAssessment(StrictModel):
    testName: str
    conclusion: str


class ObjectiveTest(StrictModel):
    testName: str
    unitName: str
    value: str
    left: str
    right: str
    comments: str


class ObjectiveAssessment(StrictModel):
    tests: list[ObjectiveTest]


class SubjectiveGoal(StrictModel):
    goalDetails: str
    targetDate: str


class ObjectiveGoal(StrictModel):
    goalName: str
    goalCategory: str
    unitName: str
    value: str
    targetDate: str


class Recommendation(StrictModel):
    sessionType: str
    sessionFrequency: str


class PatientAdvice(StrictModel):
    adviceDetails: str


class FirstAssessment(StrictModel):
    clinicalDetails: ClinicalDetails
    subjectiveAssessments: list[SubjectiveAssessment]
    objectiveAssessment: ObjectiveAssessment
    subjectiveGoals: list[SubjectiveGoal]
    objectiveGoals: list[ObjectiveGoal]
    recommendation: list[Recommendation]
    patientAdvice: PatientAdvice
