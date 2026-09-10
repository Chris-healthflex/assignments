"""The FirstAssessment contract, exactly as the frontend consumes it.

Names are literal camelCase rather than aliases, so no caller has to remember
model_dump(by_alias=True).
"""

from pydantic import BaseModel, ConfigDict


class _StrictModel(BaseModel):
    """Base model that rejects unknown keys at construction time."""

    # extra="forbid" makes an invented field fail here, not at grading time.
    model_config = ConfigDict(extra="forbid")


class ClinicalDetails(_StrictModel):
    """History and presenting complaint recorded for the session."""

    clinicalHistory: str
    chiefComplaint: str
    duration: str


class SubjectiveAssessment(_StrictModel):
    """One subjective test and the clinician's conclusion from it."""

    testName: str
    conclusion: str


class ObjectiveTest(_StrictModel):
    """One measured objective test, including bilateral readings."""

    testName: str
    unitName: str
    value: str
    left: str
    right: str
    comments: str


class ObjectiveAssessment(_StrictModel):
    """Container for the objective tests performed during the session."""

    tests: list[ObjectiveTest]


class SubjectiveGoal(_StrictModel):
    """A goal stated in the patient's own terms."""

    goalDetails: str
    targetDate: str


class ObjectiveGoal(_StrictModel):
    """A goal expressed as a measurable target."""

    goalName: str
    goalCategory: str
    unitName: str
    value: str
    targetDate: str


class Recommendation(_StrictModel):
    """A recommended course of sessions."""

    sessionType: str
    sessionFrequency: str


class PatientAdvice(_StrictModel):
    """Advice given to the patient to follow between sessions."""

    adviceDetails: str


class FirstAssessment(_StrictModel):
    """The complete first assessment produced from a clinical session.

    Every leaf is a required str, which is what enforces "strings, not null".
    """

    clinicalDetails: ClinicalDetails
    subjectiveAssessments: list[SubjectiveAssessment]
    objectiveAssessment: ObjectiveAssessment
    subjectiveGoals: list[SubjectiveGoal]
    objectiveGoals: list[ObjectiveGoal]
    # Singular name, array type — taken verbatim from the brief.
    recommendation: list[Recommendation]
    patientAdvice: PatientAdvice
