from pydantic import BaseModel, ConfigDict, Field, model_validator


class AssessmentBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="before")
    @classmethod
    def normalize_missing_strings(cls, data):
        if not isinstance(data, dict):
            return data

        normalized = {}

        for key, value in data.items():
            if value is None:
                normalized[key] = ""
            elif (
                isinstance(value, str)
                and value.strip().lower() in {"null", "none"}
            ):
                normalized[key] = ""
            else:
                normalized[key] = value

        return normalized


class ClinicalDetails(AssessmentBaseModel):
    clinicalHistory: str = ""
    chiefComplaint: str = ""
    duration: str = ""


class SubjectiveAssessment(AssessmentBaseModel):
    testName: str = ""
    conclusion: str = ""


class ObjectiveAssessmentTest(AssessmentBaseModel):
    testName: str = ""
    unitName: str = ""
    value: str = ""
    left: str = ""
    right: str = ""
    comments: str = ""


class ObjectiveAssessment(AssessmentBaseModel):
    tests: list[ObjectiveAssessmentTest] = Field(
        default_factory=list
    )


class SubjectiveGoal(AssessmentBaseModel):
    goalDetails: str = ""
    targetDate: str = ""


class ObjectiveGoal(AssessmentBaseModel):
    goalName: str = ""
    goalCategory: str = ""
    unitName: str = ""
    value: str = ""
    targetDate: str = ""


class Recommendation(AssessmentBaseModel):
    sessionType: str = ""
    sessionFrequency: str = ""


class PatientAdvice(AssessmentBaseModel):
    adviceDetails: str = ""


class FirstAssessment(AssessmentBaseModel):
    clinicalDetails: ClinicalDetails

    subjectiveAssessments: list[SubjectiveAssessment] = Field(
        default_factory=list
    )

    objectiveAssessment: ObjectiveAssessment = Field(
        default_factory=ObjectiveAssessment
    )

    subjectiveGoals: list[SubjectiveGoal] = Field(
        default_factory=list
    )

    objectiveGoals: list[ObjectiveGoal] = Field(
        default_factory=list
    )

    recommendation: list[Recommendation] = Field(
        default_factory=list
    )

    patientAdvice: PatientAdvice