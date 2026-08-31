from datetime import datetime
from typing import Any, Dict, List

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Section(BaseModel):
    """Base for the FirstAssessment tree.

    ``extra="forbid"`` keeps the output free of any field the schema does not
    define, and the validator turns a null into an empty string so no string
    field is ever serialized as ``None``.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    @field_validator("*", mode="before")
    @classmethod
    def none_to_empty(cls, value: Any) -> Any:
        return "" if value is None else value


class ClinicalDetails(Section):
    clinicalHistory: str = ""
    chiefComplaint: str = ""
    duration: str = ""


class SubjectiveAssessment(Section):
    testName: str = ""
    conclusion: str = ""


class ObjectiveTest(Section):
    testName: str = ""
    unitName: str = ""
    value: str = ""
    left: str = ""
    right: str = ""
    comments: str = ""


class ObjectiveAssessment(Section):
    tests: List[ObjectiveTest] = []


class SubjectiveGoal(Section):
    goalDetails: str = ""
    targetDate: str = ""


class ObjectiveGoal(Section):
    goalName: str = ""
    goalCategory: str = ""
    unitName: str = ""
    value: str = ""
    targetDate: str = ""


class Recommendation(Section):
    sessionType: str = ""
    sessionFrequency: str = ""


class PatientAdvice(Section):
    adviceDetails: str = ""


class FirstAssessment(Section):
    clinicalDetails: ClinicalDetails = ClinicalDetails()
    subjectiveAssessments: List[SubjectiveAssessment] = []
    objectiveAssessment: ObjectiveAssessment = ObjectiveAssessment()
    subjectiveGoals: List[SubjectiveGoal] = []
    objectiveGoals: List[ObjectiveGoal] = []
    recommendation: List[Recommendation] = []
    patientAdvice: PatientAdvice = PatientAdvice()

    @field_validator(
        "subjectiveAssessments",
        "subjectiveGoals",
        "objectiveGoals",
        "recommendation",
        mode="before",
    )
    @classmethod
    def ensure_list(cls, value: Any) -> Any:
        if value is None:
            return []
        return [value] if isinstance(value, dict) else value


class SaveAssessmentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    assessment: FirstAssessment
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AssessmentRecord(BaseModel):
    id: str
    createdAt: datetime
    assessment: FirstAssessment
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AssessmentList(BaseModel):
    count: int
    items: List[AssessmentRecord]


class ErrorResponse(BaseModel):
    code: str
    message: str
    details: List[Dict[str, Any]] = Field(default_factory=list)
