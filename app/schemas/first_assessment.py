"""FirstAssessment schema — must match Stance Health's clinician frontend exactly.

No extra fields, no renamed keys. Array fields are always arrays (even with one
item); string fields are always strings, never null.
"""

from pydantic import BaseModel, Field


def _empty_str() -> str:
    return ""


class ClinicalDetails(BaseModel):
    clinicalHistory: str = Field(default_factory=_empty_str)
    chiefComplaint: str = Field(default_factory=_empty_str)
    duration: str = Field(default_factory=_empty_str)


class SubjectiveAssessment(BaseModel):
    testName: str = Field(default_factory=_empty_str)
    conclusion: str = Field(default_factory=_empty_str)


class ObjectiveTest(BaseModel):
    testName: str = Field(default_factory=_empty_str)
    unitName: str = Field(default_factory=_empty_str)
    value: str = Field(default_factory=_empty_str)
    left: str = Field(default_factory=_empty_str)
    right: str = Field(default_factory=_empty_str)
    comments: str = Field(default_factory=_empty_str)


class ObjectiveAssessment(BaseModel):
    tests: list[ObjectiveTest] = Field(default_factory=list)


class SubjectiveGoal(BaseModel):
    goalDetails: str = Field(default_factory=_empty_str)
    targetDate: str = Field(default_factory=_empty_str)


class ObjectiveGoal(BaseModel):
    goalName: str = Field(default_factory=_empty_str)
    goalCategory: str = Field(default_factory=_empty_str)
    unitName: str = Field(default_factory=_empty_str)
    value: str = Field(default_factory=_empty_str)
    targetDate: str = Field(default_factory=_empty_str)


class Recommendation(BaseModel):
    sessionType: str = Field(default_factory=_empty_str)
    sessionFrequency: str = Field(default_factory=_empty_str)


class PatientAdvice(BaseModel):
    adviceDetails: str = Field(default_factory=_empty_str)


class FirstAssessment(BaseModel):
    clinicalDetails: ClinicalDetails = Field(default_factory=ClinicalDetails)
    subjectiveAssessments: list[SubjectiveAssessment] = Field(default_factory=list)
    objectiveAssessment: ObjectiveAssessment = Field(default_factory=ObjectiveAssessment)
    subjectiveGoals: list[SubjectiveGoal] = Field(default_factory=list)
    objectiveGoals: list[ObjectiveGoal] = Field(default_factory=list)
    recommendation: list[Recommendation] = Field(default_factory=list)
    patientAdvice: PatientAdvice = Field(default_factory=PatientAdvice)


# Top-level section names the extraction agent may flag as low-confidence.
ASSESSMENT_SECTIONS = list(FirstAssessment.model_fields.keys())
