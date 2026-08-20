from typing import Any
from pydantic import BaseModel, ConfigDict, Field, field_validator


class StrictSchemaBase(BaseModel):
    """Base model enforcing string non-nullability and forbidding extra fields."""
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        populate_by_name=True,
    )

    @field_validator("*", mode="before")
    @classmethod
    def convert_none_to_empty_or_valid(cls, v: Any) -> Any:
        if v is None:
            return ""
        return v


class ClinicalDetails(StrictSchemaBase):
    clinicalHistory: str = Field(default="", description="Relevant patient clinical history explicitly stated")
    chiefComplaint: str = Field(default="", description="Primary complaint/issue reported by patient")
    duration: str = Field(default="", description="Duration explicitly mentioned in conversation")


class SubjectiveAssessment(StrictSchemaBase):
    testName: str = Field(default="", description="Name of subjective test or inquiry")
    conclusion: str = Field(default="", description="Subjective finding or patient observation")


class ObjectiveTest(StrictSchemaBase):
    testName: str = Field(default="", description="Objective test or clinical measurement name")
    unitName: str = Field(default="", description="Measurement unit explicitly stated (e.g. degrees, kg, mmHg)")
    value: str = Field(default="", description="Recorded measurement value")
    left: str = Field(default="", description="Left side value if explicitly tested")
    right: str = Field(default="", description="Right side value if explicitly tested")
    comments: str = Field(default="", description="Objective clinical comments or notes")


class ObjectiveAssessment(StrictSchemaBase):
    tests: list[ObjectiveTest] = Field(
        default_factory=list,
        description="Array of objective clinical tests and measurements"
    )

    @field_validator("tests", mode="before")
    @classmethod
    def validate_tests_array(cls, v: Any) -> Any:
        if v is None:
            return []
        if isinstance(v, dict):
            return [v]
        return v


class SubjectiveGoal(StrictSchemaBase):
    goalDetails: str = Field(default="", description="Patient-reported functional or lifestyle goal")
    targetDate: str = Field(default="", description="Explicitly stated target date (never inferred)")


class ObjectiveGoal(StrictSchemaBase):
    goalName: str = Field(default="", description="Measurable clinical goal name")
    goalCategory: str = Field(default="", description="Goal category (e.g. Range of Motion, Strength)")
    unitName: str = Field(default="", description="Measurement unit explicitly mentioned")
    value: str = Field(default="", description="Target measurement value")
    targetDate: str = Field(default="", description="Explicitly stated target date (never inferred)")


class Recommendation(StrictSchemaBase):
    sessionType: str = Field(default="", description="Type of clinical session or therapy recommended")
    sessionFrequency: str = Field(default="", description="Frequency of sessions explicitly advised")


class PatientAdvice(StrictSchemaBase):
    adviceDetails: str = Field(default="", description="Explicit advice/instructions given to the patient")


class FirstAssessment(StrictSchemaBase):
    """
    Exact FirstAssessment schema expected by the frontend.
    Guarantees strict casing, non-null strings, and persistent array formats.
    """
    clinicalDetails: ClinicalDetails = Field(default_factory=ClinicalDetails)
    subjectiveAssessments: list[SubjectiveAssessment] = Field(default_factory=list)
    objectiveAssessment: ObjectiveAssessment = Field(default_factory=ObjectiveAssessment)
    subjectiveGoals: list[SubjectiveGoal] = Field(default_factory=list)
    objectiveGoals: list[ObjectiveGoal] = Field(default_factory=list)
    recommendation: list[Recommendation] = Field(default_factory=list)
    patientAdvice: PatientAdvice = Field(default_factory=PatientAdvice)

    @field_validator(
        "subjectiveAssessments",
        "subjectiveGoals",
        "objectiveGoals",
        "recommendation",
        mode="before"
    )
    @classmethod
    def ensure_list(cls, v: Any) -> Any:
        if v is None:
            return []
        if isinstance(v, dict):
            return [v]
        return v

    @field_validator("clinicalDetails", "objectiveAssessment", "patientAdvice", mode="before")
    @classmethod
    def ensure_dict(cls, v: Any) -> Any:
        if v is None:
            return {}
        return v
