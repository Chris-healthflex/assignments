from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.extraction import FieldExtraction


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RawClinicalDetails(_Base):
    clinicalHistory: FieldExtraction
    chiefComplaint: FieldExtraction
    duration: FieldExtraction


class RawSubjectiveAssessment(_Base):
    testName: FieldExtraction
    conclusion: FieldExtraction


class RawObjectiveTest(_Base):
    testName: FieldExtraction
    unitName: FieldExtraction
    value: FieldExtraction
    left: FieldExtraction
    right: FieldExtraction
    comments: FieldExtraction


class RawObjectiveAssessment(_Base):
    tests: list[RawObjectiveTest] = Field(default_factory=list)


class RawSubjectiveGoal(_Base):
    goalDetails: FieldExtraction
    targetDate: FieldExtraction


class RawObjectiveGoal(_Base):
    goalName: FieldExtraction
    goalCategory: FieldExtraction
    unitName: FieldExtraction
    value: FieldExtraction
    targetDate: FieldExtraction


class RawRecommendation(_Base):
    sessionType: FieldExtraction
    sessionFrequency: FieldExtraction


class RawPatientAdvice(_Base):
    adviceDetails: FieldExtraction


class RawFirstAssessment(_Base):
    """Top-level structured-output target for the LLM."""

    clinicalDetails: RawClinicalDetails
    subjectiveAssessments: list[RawSubjectiveAssessment] = Field(default_factory=list)
    objectiveAssessment: RawObjectiveAssessment
    subjectiveGoals: list[RawSubjectiveGoal] = Field(default_factory=list)
    objectiveGoals: list[RawObjectiveGoal] = Field(default_factory=list)
    recommendation: list[RawRecommendation] = Field(default_factory=list)
    patientAdvice: RawPatientAdvice
