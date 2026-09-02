"""
FirstAssessment schema (schema/v1) — the exact JSON consumed by the
Stance Health clinician frontend.

Rules enforced here:
  * key names are verbatim from the brief (camelCase, no renames)
  * `extra="forbid"` -> no additional keys can ever leak into the output
  * every string field is `str` (never None); unknown values become ""
  * every list field is a `list` (never None), possibly empty
"""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator


class _Strict(BaseModel):
    """Base: forbid extra keys, coerce None -> '' for strings."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    @field_validator("*", mode="before")
    @classmethod
    def _none_to_empty_string(cls, v, info):
        # Only strings are allowed to be missing; lists must be lists.
        field = cls.model_fields[info.field_name]
        if v is None and field.annotation is str:
            return ""
        return v


# ---- 1. clinicalDetails --------------------------------------------------
class ClinicalDetails(_Strict):
    clinicalHistory: str = ""
    chiefComplaint: str = ""
    duration: str = ""


# ---- 2. subjectiveAssessments[] -----------------------------------------
class SubjectiveAssessment(_Strict):
    testName: str = ""
    conclusion: str = ""


# ---- 3. objectiveAssessment.tests[] -------------------------------------
class ObjectiveTest(_Strict):
    testName: str = ""
    unitName: str = ""
    value: str = ""
    left: str = ""
    right: str = ""
    comments: str = ""


class ObjectiveAssessment(_Strict):
    tests: list[ObjectiveTest] = Field(default_factory=list)


# ---- 4. subjectiveGoals[] -------------------------------------------------
class SubjectiveGoal(_Strict):
    goalDetails: str = ""
    targetDate: str = ""


# ---- 5. objectiveGoals[] --------------------------------------------------
class ObjectiveGoal(_Strict):
    goalName: str = ""
    goalCategory: str = ""
    unitName: str = ""
    value: str = ""
    targetDate: str = ""


# ---- 6. recommendation[] --------------------------------------------------
class Recommendation(_Strict):
    sessionType: str = ""
    sessionFrequency: str = ""


# ---- 7. patientAdvice -----------------------------------------------------
class PatientAdvice(_Strict):
    adviceDetails: str = ""


# ---- Root -----------------------------------------------------------------
class FirstAssessment(_Strict):
    clinicalDetails: ClinicalDetails = Field(default_factory=ClinicalDetails)
    subjectiveAssessments: list[SubjectiveAssessment] = Field(default_factory=list)
    objectiveAssessment: ObjectiveAssessment = Field(default_factory=ObjectiveAssessment)
    subjectiveGoals: list[SubjectiveGoal] = Field(default_factory=list)
    objectiveGoals: list[ObjectiveGoal] = Field(default_factory=list)
    recommendation: list[Recommendation] = Field(default_factory=list)
    patientAdvice: PatientAdvice = Field(default_factory=PatientAdvice)


# ===========================================================================
# Internal (agent-only) models. These are NEVER returned to the frontend.
# The LLM fills FirstAssessment + a per-field confidence ledger; the API
# strips the ledger before responding so the payload is an exact match.
# ===========================================================================
class FieldFlag(BaseModel):
    """A field the agent could not extract with confidence."""

    field: str = Field(description="Dot path, e.g. 'clinicalDetails.duration' or 'objectiveGoals[0].targetDate'")
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str = Field(description="Why this could not be confidently extracted from the transcript")


class ExtractionDraft(BaseModel):
    """What the LLM is asked to produce (structured output target)."""

    assessment: FirstAssessment
    flags: list[FieldFlag] = Field(default_factory=list)
    overall_confidence: float = Field(ge=0.0, le=1.0, description="Confidence that the assessment faithfully reflects the transcript")


class ExtractionResult(BaseModel):
    """Final pipeline output (internal)."""

    assessment: FirstAssessment
    flags: list[FieldFlag]
    overall_confidence: float
    transcript: str
    low_confidence: bool
