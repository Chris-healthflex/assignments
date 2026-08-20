"""Pydantic models for the FirstAssessment domain.

Scaffold note: field names/enums below are a first pass and must be reconciled
with the assignment's canonical FirstAssessment spec once the repo is readable.
Everything is Optional by design -- the extraction agent fills only what the
transcript actually supports, and never invents clinical detail.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class StrictModel(BaseModel):
    """Base: reject unknown keys so schema drift fails loudly, not silently."""

    model_config = ConfigDict(extra="forbid", use_enum_values=False)


# --------------------------------------------------------------------------- #
# Enums
# --------------------------------------------------------------------------- #
class Sex(str, Enum):
    male = "male"
    female = "female"
    other = "other"
    unknown = "unknown"


class PainQuality(str, Enum):
    sharp = "sharp"
    dull = "dull"
    burning = "burning"
    aching = "aching"
    throbbing = "throbbing"
    tingling = "tingling"
    numb = "numb"
    stiff = "stiff"
    other = "other"


class Onset(str, Enum):
    sudden = "sudden"
    gradual = "gradual"
    traumatic = "traumatic"
    post_surgical = "post_surgical"
    unknown = "unknown"


class Trend(str, Enum):
    improving = "improving"
    worsening = "worsening"
    unchanged = "unchanged"
    fluctuating = "fluctuating"
    unknown = "unknown"


# --------------------------------------------------------------------------- #
# Sub-models
# --------------------------------------------------------------------------- #
class Patient(StrictModel):
    name: str | None = None
    age: int | None = Field(default=None, ge=0, le=120)
    sex: Sex | None = None
    occupation: str | None = None


class Complaint(StrictModel):
    body_region: str | None = Field(
        default=None, description="e.g. 'lower back', 'left knee'"
    )
    side: Literal["left", "right", "bilateral", "central"] | None = None
    pain_score: int | None = Field(
        default=None, ge=0, le=10, description="Patient-reported NPRS, 0-10"
    )
    quality: list[PainQuality] = Field(default_factory=list)
    onset: Onset | None = None
    duration: str | None = Field(
        default=None, description="Verbatim duration, e.g. 'about 3 weeks'"
    )
    trend: Trend | None = None
    aggravating_factors: list[str] = Field(default_factory=list)
    relieving_factors: list[str] = Field(default_factory=list)
    is_primary: bool = False


class MedicalHistory(StrictModel):
    conditions: list[str] = Field(default_factory=list)
    surgeries: list[str] = Field(default_factory=list)
    medications: list[str] = Field(default_factory=list)
    allergies: list[str] = Field(default_factory=list)
    imaging: list[str] = Field(default_factory=list)
    red_flags: list[str] = Field(
        default_factory=list,
        description="Explicitly stated warning signs (e.g. unexplained weight loss).",
    )


class Lifestyle(StrictModel):
    activity_level: str | None = None
    exercise_routine: str | None = None
    sleep_quality: str | None = None
    smoking: bool | None = None
    alcohol: str | None = None


class Goal(StrictModel):
    description: str
    timeframe: str | None = None


class ExtractionMeta(StrictModel):
    """Provenance for a single extraction run -- what produced this document."""

    model: str | None = None
    schema_version: str = "0.1.0"
    transcript_language: str | None = None
    transcript_duration_sec: float | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    unresolved_fields: list[str] = Field(
        default_factory=list, description="Fields the agent could not ground."
    )
    warnings: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Root document
# --------------------------------------------------------------------------- #
class FirstAssessment(StrictModel):
    """Structured output of one first-assessment consultation."""

    assessment_id: str | None = Field(
        default=None, description="Mongo _id as string; set on read."
    )
    session_id: str | None = None
    patient: Patient = Field(default_factory=Patient)
    complaints: list[Complaint] = Field(default_factory=list)
    medical_history: MedicalHistory = Field(default_factory=MedicalHistory)
    lifestyle: Lifestyle = Field(default_factory=Lifestyle)
    goals: list[Goal] = Field(default_factory=list)
    clinician_notes: str | None = None
    transcript: str | None = None
    meta: ExtractionMeta = Field(default_factory=ExtractionMeta)
    created_at: datetime = Field(default_factory=_utcnow)


class ExtractionPayload(StrictModel):
    """The subset of FirstAssessment the LLM is asked to produce.

    Kept separate from FirstAssessment so ids, timestamps and provenance are
    owned by our code, never by the model.
    """

    patient: Patient = Field(default_factory=Patient)
    complaints: list[Complaint] = Field(default_factory=list)
    medical_history: MedicalHistory = Field(default_factory=MedicalHistory)
    lifestyle: Lifestyle = Field(default_factory=Lifestyle)
    goals: list[Goal] = Field(default_factory=list)
    clinician_notes: str | None = None
    unresolved_fields: list[str] = Field(
        default_factory=list,
        description="Fields the transcript did not support; left unfilled on purpose.",
    )


# --------------------------------------------------------------------------- #
# API envelopes
# --------------------------------------------------------------------------- #
class TranscriptionResult(StrictModel):
    text: str
    language: str | None = None
    duration_sec: float | None = None
    segments: list[dict[str, Any]] = Field(default_factory=list)


class ExtractRequest(StrictModel):
    transcript: str = Field(min_length=1)
    session_id: str | None = None


class AssessmentResponse(StrictModel):
    assessment_id: str
    assessment: FirstAssessment
