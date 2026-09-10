"""Internal models the LLM fills, one evidence quote per leaf.

Mirrors FirstAssessment field for field, but each leaf carries the transcript
span behind it - making the model show its work is what makes grounding checkable.
"""

from pydantic import BaseModel, ConfigDict


class _StrictModel(BaseModel):
    """Base model that rejects unknown keys at construction time."""

    model_config = ConfigDict(extra="forbid")


class ExtractedValue(_StrictModel):
    """A single extracted datum and the transcript span supporting it.

    Both are "" when the transcript does not state the field.
    """

    value: str
    evidence: str


class RawClinicalDetails(_StrictModel):
    """Evidence-carrying counterpart of ``schema.ClinicalDetails``."""

    clinicalHistory: ExtractedValue
    chiefComplaint: ExtractedValue
    duration: ExtractedValue


class RawSubjectiveAssessment(_StrictModel):
    """Evidence-carrying counterpart of ``schema.SubjectiveAssessment``."""

    testName: ExtractedValue
    conclusion: ExtractedValue


class RawObjectiveTest(_StrictModel):
    """Evidence-carrying counterpart of ``schema.ObjectiveTest``."""

    testName: ExtractedValue
    unitName: ExtractedValue
    value: ExtractedValue
    left: ExtractedValue
    right: ExtractedValue
    comments: ExtractedValue


class RawObjectiveAssessment(_StrictModel):
    """Evidence-carrying counterpart of ``schema.ObjectiveAssessment``."""

    tests: list[RawObjectiveTest]


class RawSubjectiveGoal(_StrictModel):
    """Evidence-carrying counterpart of ``schema.SubjectiveGoal``."""

    goalDetails: ExtractedValue
    targetDate: ExtractedValue


class RawObjectiveGoal(_StrictModel):
    """Evidence-carrying counterpart of ``schema.ObjectiveGoal``."""

    goalName: ExtractedValue
    goalCategory: ExtractedValue
    unitName: ExtractedValue
    value: ExtractedValue
    targetDate: ExtractedValue


class RawRecommendation(_StrictModel):
    """Evidence-carrying counterpart of ``schema.Recommendation``."""

    sessionType: ExtractedValue
    sessionFrequency: ExtractedValue


class RawPatientAdvice(_StrictModel):
    """Evidence-carrying counterpart of ``schema.PatientAdvice``."""

    adviceDetails: ExtractedValue


class RawExtraction(_StrictModel):
    """The complete evidence-carrying extraction returned by the model."""

    clinicalDetails: RawClinicalDetails
    subjectiveAssessments: list[RawSubjectiveAssessment]
    objectiveAssessment: RawObjectiveAssessment
    subjectiveGoals: list[RawSubjectiveGoal]
    objectiveGoals: list[RawObjectiveGoal]
    recommendation: list[RawRecommendation]
    patientAdvice: RawPatientAdvice
