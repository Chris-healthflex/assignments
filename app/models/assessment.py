from typing import List

from pydantic import BaseModel, ConfigDict, Field


class ClinicalDetails(BaseModel):
    """
    Clinical history and presenting complaint.
    All production-facing fields are strings.
    """

    model_config = ConfigDict(extra="forbid")

    clinicalHistory: str = ""
    chiefComplaint: str = ""
    duration: str = ""


class SubjectiveAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    testName: str = ""
    conclusion: str = ""


class ObjectiveTest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    testName: str = ""
    unitName: str = ""
    value: str = ""
    left: str = ""
    right: str = ""
    comments: str = ""


class ObjectiveAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tests: List[ObjectiveTest] = Field(default_factory=list)


class SubjectiveGoal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    goalDetails: str = ""
    targetDate: str = ""


class ObjectiveGoal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    goalName: str = ""
    goalCategory: str = ""
    unitName: str = ""
    value: str = ""
    targetDate: str = ""


class Recommendation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sessionType: str = ""
    sessionFrequency: str = ""


class PatientAdvice(BaseModel):
    model_config = ConfigDict(extra="forbid")

    adviceDetails: str = ""


class FirstAssessment(BaseModel):
    """
    Exact production-facing assessment schema.

    Do not add fields here without changing the production contract.
    """

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
    )

    clinicalDetails: ClinicalDetails
    subjectiveAssessments: List[SubjectiveAssessment] = Field(
        default_factory=list
    )
    objectiveAssessment: ObjectiveAssessment
    subjectiveGoals: List[SubjectiveGoal] = Field(
        default_factory=list
    )
    objectiveGoals: List[ObjectiveGoal] = Field(
        default_factory=list
    )
    recommendation: List[Recommendation] = Field(
        default_factory=list
    )
    patientAdvice: PatientAdvice
