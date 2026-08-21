from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


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


class ObjectiveAssessment(StrictModel):
    tests: list[ObjectiveTest] = Field(default_factory=list)


class FirstAssessment(StrictModel):
    clinicalDetails: ClinicalDetails = Field(default_factory=ClinicalDetails)
    subjectiveAssessments: list[SubjectiveAssessment] = Field(default_factory=list)
    objectiveAssessment: ObjectiveAssessment = Field(default_factory=ObjectiveAssessment)
    subjectiveGoals: list[SubjectiveGoal] = Field(default_factory=list)
    objectiveGoals: list[ObjectiveGoal] = Field(default_factory=list)
    recommendation: list[Recommendation] = Field(default_factory=list)
    patientAdvice: PatientAdvice = Field(default_factory=PatientAdvice)
