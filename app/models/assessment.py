from __future__ import annotations
from typing import Any, List, get_origin
from pydantic import BaseModel, ConfigDict, Field, field_validator


class _Strict(BaseModel):
    """Shared contract rules: no unknown keys, no null strings, no null lists/models."""
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    @field_validator("*", mode="before")
    @classmethod
    def _coerce_nulls(cls, v: Any, info):
        field = cls.model_fields.get(info.field_name)
        if field is not None and v is None:
            if field.annotation is str:
                return ""
            origin = get_origin(field.annotation)
            if origin is list or field.annotation is list:
                return []
            if isinstance(field.annotation, type) and issubclass(field.annotation, BaseModel):
                return field.annotation()
        return v


class ClinicalDetails(_Strict):
    clinicalHistory: str = ""
    chiefComplaint: str = ""
    duration: str = ""


class SubjectiveAssessment(_Strict):
    testName: str = ""
    conclusion: str = ""


class ObjectiveTest(_Strict):
    testName: str = ""
    unitName: str = ""
    value: str = ""
    left: str = ""
    right: str = ""
    comments: str = ""


class ObjectiveAssessment(_Strict):
    tests: List[ObjectiveTest] = Field(default_factory=list)


class SubjectiveGoal(_Strict):
    goalDetails: str = ""
    targetDate: str = ""


class ObjectiveGoal(_Strict):
    goalName: str = ""
    goalCategory: str = ""
    unitName: str = ""
    value: str = ""
    targetDate: str = ""


class Recommendation(_Strict):
    sessionType: str = ""
    sessionFrequency: str = ""


class PatientAdvice(_Strict):
    adviceDetails: str = ""


class FirstAssessment(_Strict):
    clinicalDetails: ClinicalDetails = Field(default_factory=ClinicalDetails)
    subjectiveAssessments: List[SubjectiveAssessment] = Field(default_factory=list)
    objectiveAssessment: ObjectiveAssessment = Field(default_factory=ObjectiveAssessment)
    subjectiveGoals: List[SubjectiveGoal] = Field(default_factory=list)
    objectiveGoals: List[ObjectiveGoal] = Field(default_factory=list)
    recommendation: List[Recommendation] = Field(default_factory=list)
    patientAdvice: PatientAdvice = Field(default_factory=PatientAdvice)
