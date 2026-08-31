"""
FirstAssessment schema — Pydantic v2.

This is the exact contract consumed by Stance Health's clinician frontend.
No extra fields, no renamed keys, no nulls in string fields, arrays are
always arrays (even length 1, even empty — never null/omitted).

Every field carries a `confidence` shadow entry (see ExtractionResult below)
rather than baking confidence into the schema itself — the frontend schema
stays pristine, and low-confidence fields are reported alongside it, not
inside it.
"""
from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator


class StrictModel(BaseModel):
    """Base class: no undeclared fields ever leak into the frontend payload."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


# ---------------------------------------------------------------------------
# 1. clinicalDetails
# ---------------------------------------------------------------------------
class ClinicalDetails(StrictModel):
    clinicalHistory: str = ""
    chiefComplaint: str = ""
    duration: str = ""


# ---------------------------------------------------------------------------
# 2. subjectiveAssessments[]
# ---------------------------------------------------------------------------
class SubjectiveAssessment(StrictModel):
    testName: str
    conclusion: str


# ---------------------------------------------------------------------------
# 3. objectiveAssessment.tests[]
# ---------------------------------------------------------------------------
class ObjectiveTest(StrictModel):
    testName: str
    unitName: str = ""
    value: str = ""
    left: str = ""
    right: str = ""
    comments: str = ""


class ObjectiveAssessment(StrictModel):
    tests: List[ObjectiveTest] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# 4. subjectiveGoals[]
# ---------------------------------------------------------------------------
class SubjectiveGoal(StrictModel):
    goalDetails: str
    targetDate: str = ""  # ISO 8601 (YYYY-MM-DD) when known, else ""


# ---------------------------------------------------------------------------
# 5. objectiveGoals[]
# ---------------------------------------------------------------------------
class ObjectiveGoal(StrictModel):
    goalName: str
    goalCategory: str = ""
    unitName: str = ""
    value: str = ""
    targetDate: str = ""


# ---------------------------------------------------------------------------
# 6. recommendation[]
# ---------------------------------------------------------------------------
class Recommendation(StrictModel):
    sessionType: str
    sessionFrequency: str = ""


# ---------------------------------------------------------------------------
# 7. patientAdvice
# ---------------------------------------------------------------------------
class PatientAdvice(StrictModel):
    adviceDetails: str = ""


# ---------------------------------------------------------------------------
# Top-level FirstAssessment
# ---------------------------------------------------------------------------
class FirstAssessment(StrictModel):
    clinicalDetails: ClinicalDetails = Field(default_factory=ClinicalDetails)
    subjectiveAssessments: List[SubjectiveAssessment] = Field(default_factory=list)
    objectiveAssessment: ObjectiveAssessment = Field(default_factory=ObjectiveAssessment)
    subjectiveGoals: List[SubjectiveGoal] = Field(default_factory=list)
    objectiveGoals: List[ObjectiveGoal] = Field(default_factory=list)
    recommendation: List[Recommendation] = Field(default_factory=list)
    patientAdvice: PatientAdvice = Field(default_factory=PatientAdvice)

    @field_validator(
        "subjectiveAssessments",
        "objectiveGoals",
        "subjectiveGoals",
        "recommendation",
        mode="before",
    )
    @classmethod
    def _never_none(cls, v):
        # Guards against the LLM emitting `null` for an array field instead
        # of `[]` — the brief is explicit that arrays must always be arrays.
        return v if v is not None else []


# ---------------------------------------------------------------------------
# Extraction metadata — NOT part of the frontend payload.
# Returned separately by /assessments/parse alongside `assessment` so low
# confidence fields can trigger a 422 without polluting the strict schema.
# ---------------------------------------------------------------------------
class FieldConfidence(StrictModel):
    field_path: str  # dot-path, e.g. "clinicalDetails.chiefComplaint"
    confidence: float  # 0.0 - 1.0
    reason: Optional[str] = None


class ExtractionResult(StrictModel):
    assessment: FirstAssessment
    transcript: str
    field_confidences: List[FieldConfidence] = Field(default_factory=list)
    overall_confidence: float = 1.0
    low_confidence_fields: List[str] = Field(default_factory=list)
