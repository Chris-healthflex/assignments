"""Strict Pydantic v2 schemas for FirstAssessment production output.

Constraints:
- ConfigDict(extra="forbid") on all models to prevent extra/undocumented keys.
- All array fields default to factory list and serialize as lists (even when empty or single-element).
- All string fields default to empty string "" and never serialize to None/null.
- clinicalDetails.duration is modeled as a generic dictionary (Dict[str, Any]) because the assignment
  specification defines duration only as an object without documenting internal child keys.
- objectiveAssessment contains exclusively tests[] with no top-level comments field.
- subjectiveAssessments[].conclusion is defined as a list of strings (List[str]).
"""

from typing import Any, Dict, List
from pydantic import BaseModel, ConfigDict, Field


class StrictBaseModel(BaseModel):
    """Base model enforcing strict schema validation (extra fields are forbidden)."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        populate_by_name=True,
    )


class ClinicalDetails(StrictBaseModel):
    """Clinical details including history, chief complaint, and duration."""

    clinicalHistory: str = ""
    chiefComplaint: str = ""
    # Note: The assignment specification identifies duration as an object (`obj`)
    # without exposing or documenting its internal child keys. We intentionally
    # represent it as a generic dictionary object rather than inventing unconfirmed keys.
    duration: Dict[str, Any] = Field(default_factory=dict)


class SubjectiveAssessment(StrictBaseModel):
    """Subjective assessment test with conclusion list."""

    testName: str = ""
    conclusion: List[str] = Field(default_factory=list)


class ObjectiveTest(StrictBaseModel):
    """Individual objective test with measurements and comments."""

    testName: str = ""
    unitName: str = ""
    value: str = ""
    left: str = ""
    right: str = ""
    comments: List[str] = Field(default_factory=list)


class ObjectiveAssessment(StrictBaseModel):
    """Objective assessment containing only tests list."""

    tests: List[ObjectiveTest] = Field(default_factory=list)


class SubjectiveGoal(StrictBaseModel):
    """Subjective clinical goal."""

    goalDetails: str = ""
    targetDate: str = ""


class ObjectiveGoal(StrictBaseModel):
    """Objective clinical goal with category, unit, value, and target date."""

    goalName: str = ""
    goalCategory: str = ""
    unitName: str = ""
    value: str = ""
    targetDate: str = ""


class Recommendation(StrictBaseModel):
    """Clinical recommendation with session type and frequency."""

    sessionType: str = ""
    sessionFrequency: str = ""


class PatientAdvice(StrictBaseModel):
    """Patient advice details."""

    adviceDetails: str = ""


class FirstAssessment(StrictBaseModel):
    """Root production schema for First Assessment output."""

    clinicalDetails: ClinicalDetails = Field(default_factory=ClinicalDetails)
    subjectiveAssessments: List[SubjectiveAssessment] = Field(default_factory=list)
    objectiveAssessment: ObjectiveAssessment = Field(default_factory=ObjectiveAssessment)
    subjectiveGoals: List[SubjectiveGoal] = Field(default_factory=list)
    objectiveGoals: List[ObjectiveGoal] = Field(default_factory=list)
    recommendation: List[Recommendation] = Field(default_factory=list)
    patientAdvice: PatientAdvice = Field(default_factory=PatientAdvice)
