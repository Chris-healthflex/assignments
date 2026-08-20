from __future__ import annotations
from typing import Annotated, Any, List
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field


def _none_to_empty(v: Any) -> Any:
    if v is None:
        return ""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return str(v)
    return v


NoneToEmptyStr = Annotated[str, BeforeValidator(_none_to_empty)]


class _Strict(BaseModel):
    """Base model: reject unknown keys so schema drift fails loudly."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class ClinicalDetails(_Strict):
    clinicalHistory: NoneToEmptyStr = ""
    chiefComplaint: NoneToEmptyStr = ""
    duration: NoneToEmptyStr = ""


class SubjectiveAssessment(_Strict):
    testName: NoneToEmptyStr = ""
    conclusion: NoneToEmptyStr = ""


class ObjectiveTest(_Strict):
    testName: NoneToEmptyStr = ""
    unitName: NoneToEmptyStr = ""
    value: NoneToEmptyStr = ""
    left: NoneToEmptyStr = ""
    right: NoneToEmptyStr = ""
    comments: NoneToEmptyStr = ""


class ObjectiveAssessment(_Strict):
    tests: List[ObjectiveTest] = Field(default_factory=list)


class SubjectiveGoal(_Strict):
    goalDetails: NoneToEmptyStr = ""
    targetDate: NoneToEmptyStr = ""


class ObjectiveGoal(_Strict):
    goalName: NoneToEmptyStr = ""
    goalCategory: NoneToEmptyStr = ""
    unitName: NoneToEmptyStr = ""
    value: NoneToEmptyStr = ""
    targetDate: NoneToEmptyStr = ""


class Recommendation(_Strict):
    sessionType: NoneToEmptyStr = ""
    sessionFrequency: NoneToEmptyStr = ""


class PatientAdvice(_Strict):
    adviceDetails: NoneToEmptyStr = ""


class FirstAssessment(_Strict):
    """The 7-section clinical assessment. This is the whole contract."""

    clinicalDetails: ClinicalDetails = Field(default_factory=ClinicalDetails)
    subjectiveAssessments: List[SubjectiveAssessment] = Field(default_factory=list)
    objectiveAssessment: ObjectiveAssessment = Field(default_factory=ObjectiveAssessment)
    subjectiveGoals: List[SubjectiveGoal] = Field(default_factory=list)
    objectiveGoals: List[ObjectiveGoal] = Field(default_factory=list)
    recommendation: List[Recommendation] = Field(default_factory=list)
    patientAdvice: PatientAdvice = Field(default_factory=PatientAdvice)


# Dotted paths of every leaf the extractor is expected to fill on a scalar
# section. Used to compute "what did we fail to extract" without hardcoding
# strings in two places.
SCALAR_FIELD_PATHS: tuple[str, ...] = (
    "clinicalDetails.clinicalHistory",
    "clinicalDetails.chiefComplaint",
    "clinicalDetails.duration",
    "patientAdvice.adviceDetails",
)

LIST_FIELD_PATHS: tuple[str, ...] = (
    "subjectiveAssessments",
    "objectiveAssessment.tests",
    "subjectiveGoals",
    "objectiveGoals",
    "recommendation",
)
