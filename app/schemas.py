"""Strict production schema for a first clinical assessment."""
from __future__ import annotations

from typing import List

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
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
    tests: List[ObjectiveTest]


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
    subjectiveAssessments: List[SubjectiveAssessment]
    objectiveAssessment: ObjectiveAssessment
    subjectiveGoals: List[SubjectiveGoal]
    objectiveGoals: List[ObjectiveGoal]
    recommendation: List[Recommendation]
    patientAdvice: PatientAdvice


class FieldIssue(StrictModel):
    field: str
    reason: str


class ExtractionEnvelope(StrictModel):
    """Internal agent response. It is never exposed by the parse endpoint."""

    assessment: FirstAssessment
    uncertain_fields: List[FieldIssue] = Field(default_factory=list)


class SavedAssessment(FirstAssessment):
    id: str
    createdAt: str
