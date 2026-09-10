from pydantic import BaseModel, ConfigDict


class Duration(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: str
    unit: str


class ClinicalDetails(BaseModel):
    model_config = ConfigDict(extra="forbid")

    clinicalHistory: str
    chiefComplaint: str
    duration: Duration


class SubjectiveAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    testName: str
    conclusion: list[str]


class ObjectiveTest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    testName: str
    unitName: str
    value: str
    left: str
    right: str
    comments: list[str]


class ObjectiveAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tests: list[ObjectiveTest]


class SubjectiveGoal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    goalDetails: str
    targetDate: list[str]


class ObjectiveGoal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    goalName: str
    goalCategory: str
    unitName: str
    value: str
    targetDate: list[str]


class Recommendation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sessionType: str
    sessionFrequency: str


class PatientAdvice(BaseModel):
    model_config = ConfigDict(extra="forbid")

    adviceDetails: dict[str, str]


class FirstAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    clinicalDetails: ClinicalDetails
    subjectiveAssessments: list[SubjectiveAssessment]
    objectiveAssessment: ObjectiveAssessment
    subjectiveGoals: list[SubjectiveGoal]
    objectiveGoals: list[ObjectiveGoal]
    recommendation: list[Recommendation]
    patientAdvice: PatientAdvice