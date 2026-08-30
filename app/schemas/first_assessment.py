from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    @field_validator("*", mode="before")
    @classmethod
    def normalize_missing_strings(cls, value: Any, info: Any) -> Any:
        if cls.model_fields[info.field_name].annotation is str and value is None:
            return ""
        return value


class ClinicalDetails(StrictModel):
    clinicalHistory: str = ""
    chiefComplaint: str = ""
    duration: str = ""


class SubjectiveAssessment(StrictModel):
    testName: str = ""
    conclusion: str = ""


class ObjectiveTest(StrictModel):
    testName: str = ""
    unitName: str = ""
    value: str = ""
    left: str = ""
    right: str = ""
    comments: str = ""


class ObjectiveAssessment(StrictModel):
    tests: list[ObjectiveTest] = Field(default_factory=list)


class SubjectiveGoal(StrictModel):
    goalDetails: str = ""
    targetDate: str = ""


class ObjectiveGoal(StrictModel):
    goalName: str = ""
    goalCategory: str = ""
    unitName: str = ""
    value: str = ""
    targetDate: str = ""


class Recommendation(StrictModel):
    sessionType: str = ""
    sessionFrequency: str = ""


class PatientAdvice(StrictModel):
    adviceDetails: str = ""


class FirstAssessment(StrictModel):
    clinicalDetails: ClinicalDetails = Field(default_factory=ClinicalDetails)
    subjectiveAssessments: list[SubjectiveAssessment] = Field(default_factory=list)
    objectiveAssessment: ObjectiveAssessment = Field(default_factory=ObjectiveAssessment)
    subjectiveGoals: list[SubjectiveGoal] = Field(default_factory=list)
    objectiveGoals: list[ObjectiveGoal] = Field(default_factory=list)
    recommendation: list[Recommendation] = Field(default_factory=list)
    patientAdvice: PatientAdvice = Field(default_factory=PatientAdvice)

    @model_validator(mode="before")
    @classmethod
    def normalize_missing_sections(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value

        data = dict(value)
        object_defaults = {
            "clinicalDetails": {},
            "objectiveAssessment": {},
            "patientAdvice": {},
        }
        list_defaults = {
            "subjectiveAssessments": [],
            "subjectiveGoals": [],
            "objectiveGoals": [],
            "recommendation": [],
        }

        for key, default in object_defaults.items():
            if data.get(key) is None:
                data[key] = default

        for key, default in list_defaults.items():
            if data.get(key) is None:
                data[key] = default

        return data

    @field_validator(
        "subjectiveAssessments",
        "subjectiveGoals",
        "objectiveGoals",
        "recommendation",
        mode="before",
    )
    @classmethod
    def normalize_missing_lists(cls, value: Any) -> Any:
        return [] if value is None else value


def normalize_assessment(data: dict[str, Any]) -> FirstAssessment:
    try:
        return FirstAssessment.model_validate(data)
    except ValidationError:
        raise
