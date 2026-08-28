"""
FirstAssessment schema — the exact contract the production frontend consumes.

Design rules enforced here (per the brief):
  - No extra fields, no renamed keys.
  - All array fields are arrays, even with a single item (never a bare object, never null).
  - All string fields are strings, never null (use "" or "unspecified" instead of None).
  - Every leaf value that the extraction agent could not confidently pull from the
    transcript is still populated (schema-valid), but is *also* listed in the
    sibling `extraction_flags` block on the wrapping envelope so downstream
    reviewers know which fields are low-confidence / not stated in the audio.
"""

from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field, field_validator


def _none_to_empty_str(v):
    return "" if v is None else v


class ClinicalDetails(BaseModel):
    clinicalHistory: str = ""
    chiefComplaint: str = ""
    duration: str = ""

    _norm = field_validator(
        "clinicalHistory", "chiefComplaint", "duration", mode="before"
    )(_none_to_empty_str)


class SubjectiveAssessment(BaseModel):
    testName: str = ""
    conclusion: str = ""

    _norm = field_validator("testName", "conclusion", mode="before")(_none_to_empty_str)


class ObjectiveTest(BaseModel):
    testName: str = ""
    unitName: str = ""
    value: str = ""
    left: str = ""
    right: str = ""
    comments: str = ""

    _norm = field_validator(
        "testName", "unitName", "value", "left", "right", "comments", mode="before"
    )(_none_to_empty_str)


class ObjectiveAssessment(BaseModel):
    tests: List[ObjectiveTest] = Field(default_factory=list)


class SubjectiveGoal(BaseModel):
    goalDetails: str = ""
    targetDate: str = ""

    _norm = field_validator("goalDetails", "targetDate", mode="before")(_none_to_empty_str)


class ObjectiveGoal(BaseModel):
    goalName: str = ""
    goalCategory: str = ""
    unitName: str = ""
    value: str = ""
    targetDate: str = ""

    _norm = field_validator(
        "goalName", "goalCategory", "unitName", "value", "targetDate", mode="before"
    )(_none_to_empty_str)


class Recommendation(BaseModel):
    sessionType: str = ""
    sessionFrequency: str = ""

    _norm = field_validator("sessionType", "sessionFrequency", mode="before")(
        _none_to_empty_str
    )


class PatientAdvice(BaseModel):
    adviceDetails: str = ""

    _norm = field_validator("adviceDetails", mode="before")(_none_to_empty_str)


class FirstAssessment(BaseModel):
    """Exact schema — field names and nesting must match the frontend contract."""

    clinicalDetails: ClinicalDetails = Field(default_factory=ClinicalDetails)
    subjectiveAssessments: List[SubjectiveAssessment] = Field(default_factory=list)
    objectiveAssessment: ObjectiveAssessment = Field(default_factory=ObjectiveAssessment)
    subjectiveGoals: List[SubjectiveGoal] = Field(default_factory=list)
    objectiveGoals: List[ObjectiveGoal] = Field(default_factory=list)
    recommendation: List[Recommendation] = Field(default_factory=list)
    patientAdvice: PatientAdvice = Field(default_factory=PatientAdvice)

    model_config = {"extra": "forbid"}


class ExtractionEnvelope(BaseModel):
    """
    What /assessments/parse actually returns: the schema-exact `assessment`
    payload, plus out-of-band metadata the frontend can ignore but the API/DB
    layer needs — confidence + which fields were not confidently extracted.
    This wrapper is NOT the FirstAssessment object itself; see README for
    why the two are kept separate rather than polluting the frontend schema
    with extraction metadata.
    """

    assessment: FirstAssessment
    overall_confidence: float = Field(ge=0.0, le=1.0)
    extraction_flags: List[str] = Field(
        default_factory=list,
        description="Dot-paths of fields the agent could not confidently extract "
        "from the transcript, e.g. 'clinicalDetails.duration', "
        "'objectiveGoals[1].targetDate'.",
    )
    transcript: str = ""
