from pydantic import BaseModel, ConfigDict


class ClinicalDetails(BaseModel):
    model_config = ConfigDict(extra="forbid")

    clinicalHistory: str
    chiefComplaint: str
    duration: str


class SubjectiveAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    testName: str
    conclusion: str


class ObjectiveTest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    testName: str
    unitName: str
    value: str
    left: str
    right: str
    comments: str


class ObjectiveAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tests: list[ObjectiveTest]


class SubjectiveGoal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    goalDetails: str
    targetDate: str


class ObjectiveGoal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    goalName: str
    goalCategory: str
    unitName: str
    value: str
    targetDate: str


class Recommendation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sessionType: str
    sessionFrequency: str


class PatientAdvice(BaseModel):
    model_config = ConfigDict(extra="forbid")

    adviceDetails: str


class FirstAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    clinicalDetails: ClinicalDetails
    subjectiveAssessments: list[SubjectiveAssessment]
    objectiveAssessment: ObjectiveAssessment
    subjectiveGoals: list[SubjectiveGoal]
    objectiveGoals: list[ObjectiveGoal]
    recommendation: list[Recommendation]
    patientAdvice: PatientAdvice
