"""The FirstAssessment schema consumed by the Stance Health clinician frontend.

The brief states three contract rules. They are enforced structurally here
rather than by convention, because a violation breaks the live frontend:

1. No extra fields, no renamed keys -> ``extra="forbid"`` on every model, so a
   typo like ``chiefComplaints`` raises instead of silently passing through.
2. Array fields are always arrays   -> ``None`` is coerced to ``[]``.
3. String fields are strings, never -> ``None`` is coerced to ``""``.
   null

Rule 3 carries the most weight in this pipeline. "Could not be confidently
extracted" is represented by the empty string - never by ``null``, and never by
a plausible-looking guess. Which fields ended up blank is reported separately
by ``app.extraction.confidence`` so that this model stays exactly seven keys
wide, as the frontend requires.
"""

from __future__ import annotations

from typing import Any, Iterator, List, get_origin

from pydantic import BaseModel, ConfigDict, field_validator


class _Section(BaseModel):
    """Base for every node in the schema: strict keys, no nulls, trimmed strings."""

    model_config = ConfigDict(
        extra="forbid",             # rule 1
        str_strip_whitespace=True,
        validate_assignment=True,
    )

    @field_validator("*", mode="before")
    @classmethod
    def _normalise(cls, value: Any, info) -> Any:
        """Coerce nulls and stray scalars into the field's declared shape.

        An LLM that cannot find a value will happily emit ``null``, and one that
        reads "120 degrees" may emit the number ``120``. The frontend expects
        ``""`` and ``"120"`` respectively. Normalising at the boundary means no
        downstream node has to defend against either case.
        """
        annotation = cls.model_fields[info.field_name].annotation
        is_list = get_origin(annotation) is list

        if value is None:
            if is_list:
                return []                      # rule 2
            if annotation is str:
                return ""                      # rule 3
            return {}                          # nested section -> its defaults

        # Numeric scalars where a string is declared: keep the value, fix the type.
        if annotation is str and isinstance(value, (int, float)) and not isinstance(value, bool):
            return str(value)

        # A single object where a list is declared. The brief is explicit that
        # array fields stay arrays "even if only one item is present", so wrap
        # rather than reject.
        if is_list and isinstance(value, dict):
            return [value]

        return value


# --------------------------------------------------------------------------
# Section 1/7 - clinicalDetails
# --------------------------------------------------------------------------
class ClinicalDetails(_Section):
    clinicalHistory: str = ""
    chiefComplaint: str = ""
    duration: str = ""


# --------------------------------------------------------------------------
# Section 2/7 - subjectiveAssessments[]
# --------------------------------------------------------------------------
class SubjectiveAssessment(_Section):
    testName: str = ""
    conclusion: str = ""


# --------------------------------------------------------------------------
# Section 3/7 - objectiveAssessment.tests[]
# --------------------------------------------------------------------------
class ObjectiveTest(_Section):
    testName: str = ""
    unitName: str = ""
    value: str = ""
    left: str = ""
    right: str = ""
    comments: str = ""


class ObjectiveAssessment(_Section):
    tests: List[ObjectiveTest] = []


# --------------------------------------------------------------------------
# Section 4/7 - subjectiveGoals[]
# --------------------------------------------------------------------------
class SubjectiveGoal(_Section):
    goalDetails: str = ""
    targetDate: str = ""


# --------------------------------------------------------------------------
# Section 5/7 - objectiveGoals[]
# --------------------------------------------------------------------------
class ObjectiveGoal(_Section):
    goalName: str = ""
    goalCategory: str = ""
    unitName: str = ""
    value: str = ""
    targetDate: str = ""


# --------------------------------------------------------------------------
# Section 6/7 - recommendation[]
# --------------------------------------------------------------------------
class Recommendation(_Section):
    sessionType: str = ""
    sessionFrequency: str = ""


# --------------------------------------------------------------------------
# Section 7/7 - patientAdvice
# --------------------------------------------------------------------------
class PatientAdvice(_Section):
    adviceDetails: str = ""


class FirstAssessment(_Section):
    """The complete assessment. Serialises to exactly these seven keys."""

    clinicalDetails: ClinicalDetails = ClinicalDetails()
    subjectiveAssessments: List[SubjectiveAssessment] = []
    objectiveAssessment: ObjectiveAssessment = ObjectiveAssessment()
    subjectiveGoals: List[SubjectiveGoal] = []
    objectiveGoals: List[ObjectiveGoal] = []
    recommendation: List[Recommendation] = []
    patientAdvice: PatientAdvice = PatientAdvice()


#: The seven top-level keys, in frontend order. Used by the contract test.
SECTION_KEYS: tuple[str, ...] = tuple(FirstAssessment.model_fields)


def empty_assessment() -> FirstAssessment:
    """A fully-formed assessment with every field blank.

    This is the pipeline's failure mode: when nothing can be grounded in the
    transcript, the caller still receives a schema-valid document rather than
    an error or a set of invented values.
    """
    return FirstAssessment()


def iter_field_paths(model: BaseModel, prefix: str = "") -> Iterator[tuple[str, str]]:
    """Yield ``(dotted_path, value)`` for every leaf string in the assessment.

    Paths look like ``clinicalDetails.chiefComplaint`` and
    ``objectiveAssessment.tests[0].value``. Phase 3 uses these to report which
    fields were flagged, and the API uses them for field-level 422 detail -
    both need to name a field precisely without re-deriving the structure.
    """
    for name, _field in type(model).model_fields.items():
        value = getattr(model, name)
        path = f"{prefix}{name}"

        if isinstance(value, BaseModel):
            yield from iter_field_paths(value, prefix=f"{path}.")
        elif isinstance(value, list):
            for index, item in enumerate(value):
                if isinstance(item, BaseModel):
                    yield from iter_field_paths(item, prefix=f"{path}[{index}].")
        else:
            yield path, value


def blank_field_paths(model: BaseModel) -> list[str]:
    """Leaf paths whose value is blank - the raw material for S5 flagging."""
    return [path for path, value in iter_field_paths(model) if not str(value).strip()]
