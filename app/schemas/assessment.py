from pydantic import BaseModel, ConfigDict, Field

class ClinicalDetails(BaseModel):
    model_config = ConfigDict(extra='forbid', str_strip_whitespace=True)
    clinicalHistory: str = ""
    chiefComplaint: str = ""
    duration: str = ""

class SubjectiveAssessment(BaseModel):
    model_config = ConfigDict(extra='forbid', str_strip_whitespace=True)
    testName: str = ""
    conclusion: str = ""

class ObjectiveTest(BaseModel):
    model_config = ConfigDict(extra='forbid', str_strip_whitespace=True)
    testName: str = ""
    unitName: str = ""
    value: str = ""
    left: str = ""
    right: str = ""
    comments: str = ""

class ObjectiveAssessment(BaseModel):
    model_config = ConfigDict(extra='forbid', str_strip_whitespace=True)
    tests: list[ObjectiveTest] = Field(default_factory=list)

class SubjectiveGoal(BaseModel):
    model_config = ConfigDict(extra='forbid', str_strip_whitespace=True)
    goalDetails: str = ""
    targetDate: str = ""

class ObjectiveGoal(BaseModel):
    model_config = ConfigDict(extra='forbid', str_strip_whitespace=True)
    goalName: str = ""
    goalCategory: str = ""
    unitName: str = ""
    value: str = ""
    targetDate: str = ""

class Recommendation(BaseModel):
    model_config = ConfigDict(extra='forbid', str_strip_whitespace=True)
    sessionType: str = ""
    sessionFrequency: str = ""

class PatientAdvice(BaseModel):
    model_config = ConfigDict(extra='forbid', str_strip_whitespace=True)
    adviceDetails: str = ""

class FirstAssessment(BaseModel):
    model_config = ConfigDict(extra='forbid', str_strip_whitespace=True)
    clinicalDetails: ClinicalDetails = Field(default_factory=ClinicalDetails)
    subjectiveAssessments: list[SubjectiveAssessment] = Field(default_factory=list)
    objectiveAssessment: ObjectiveAssessment = Field(default_factory=ObjectiveAssessment)
    subjectiveGoals: list[SubjectiveGoal] = Field(default_factory=list)
    objectiveGoals: list[ObjectiveGoal] = Field(default_factory=list)
    recommendation: list[Recommendation] = Field(default_factory=list)
    patientAdvice: PatientAdvice = Field(default_factory=PatientAdvice)