"""The `FirstAssessment` output contract and its seven sections.

Field names are camelCase to match the target schema exactly - no aliases, no
renaming, no extra fields.
"""

from typing import List

from pydantic import BaseModel, Field, field_validator

from app.schemas.validators import EmptyStr, as_list, as_object


class ClinicalDetails(BaseModel):
    clinicalHistory: EmptyStr = Field(
        default="",
        description="Patient's relevant medical history, previous injuries, or existing conditions."
    )
    chiefComplaint: EmptyStr = Field(
        default="",
        description="Primary reason for patient visit or main complaint."
    )
    duration: EmptyStr = Field(
        default="",
        description="Onset or duration of current symptoms/complaint."
    )


class SubjectiveAssessment(BaseModel):
    testName: EmptyStr = Field(
        default="",
        description="Name of subjective assessment or reported test."
    )
    conclusion: EmptyStr = Field(
        default="",
        description="Patient's subjective report or clinician's conclusion regarding the test."
    )


class ObjectiveTest(BaseModel):
    testName: EmptyStr = Field(
        default="",
        description="Name of objective test or measurement (e.g. Range of Motion, MMT, Pain Scale)."
    )
    unitName: EmptyStr = Field(
        default="",
        description="Unit of measurement (e.g. degrees, /10, kg)."
    )
    value: EmptyStr = Field(
        default="",
        description="Measured value or summary score."
    )
    left: EmptyStr = Field(
        default="",
        description="Measurement or value for left side if applicable."
    )
    right: EmptyStr = Field(
        default="",
        description="Measurement or value for right side if applicable."
    )
    comments: EmptyStr = Field(
        default="",
        description="Additional observations or comments for this test."
    )


class ObjectiveAssessment(BaseModel):
    tests: List[ObjectiveTest] = Field(
        default_factory=list,
        description="List of objective clinical tests and measurements."
    )

    @field_validator("tests", mode="before")
    @classmethod
    def ensure_list(cls, v):
        return as_list(v)


class SubjectiveGoal(BaseModel):
    goalDetails: EmptyStr = Field(
        default="",
        description="Details of patient-reported goal or functional target."
    )
    targetDate: EmptyStr = Field(
        default="",
        description="Target timeframe or date for achieving goal."
    )


class ObjectiveGoal(BaseModel):
    goalName: EmptyStr = Field(
        default="",
        description="Objective measurable goal name."
    )
    goalCategory: EmptyStr = Field(
        default="",
        description="Category of objective goal (e.g. Range of Motion, Strength, Function)."
    )
    unitName: EmptyStr = Field(
        default="",
        description="Unit of measure for goal (e.g. degrees, /10)."
    )
    value: EmptyStr = Field(
        default="",
        description="Target quantitative value."
    )
    targetDate: EmptyStr = Field(
        default="",
        description="Target timeframe or date for completion."
    )


class Recommendation(BaseModel):
    sessionType: EmptyStr = Field(
        default="",
        description="Recommended therapy or session type (e.g. Physical Therapy, Follow-up)."
    )
    sessionFrequency: EmptyStr = Field(
        default="",
        description="Frequency of sessions (e.g. 2x/week for 4 weeks)."
    )


class PatientAdvice(BaseModel):
    adviceDetails: EmptyStr = Field(
        default="",
        description="Instructions, home exercise guidance, or advice provided to the patient."
    )


class FirstAssessment(BaseModel):
    clinicalDetails: ClinicalDetails = Field(default_factory=ClinicalDetails)
    subjectiveAssessments: List[SubjectiveAssessment] = Field(default_factory=list)
    objectiveAssessment: ObjectiveAssessment = Field(default_factory=ObjectiveAssessment)
    subjectiveGoals: List[SubjectiveGoal] = Field(default_factory=list)
    objectiveGoals: List[ObjectiveGoal] = Field(default_factory=list)
    recommendation: List[Recommendation] = Field(default_factory=list)
    patientAdvice: PatientAdvice = Field(default_factory=PatientAdvice)

    @field_validator(
        "subjectiveAssessments",
        "subjectiveGoals",
        "objectiveGoals",
        "recommendation",
        mode="before",
    )
    @classmethod
    def ensure_list(cls, v):
        return as_list(v)

    @field_validator(
        "clinicalDetails", "objectiveAssessment", "patientAdvice", mode="before"
    )
    @classmethod
    def ensure_object(cls, v):
        return as_object(v)
