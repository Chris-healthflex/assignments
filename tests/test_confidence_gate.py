from app.agents.assessment_graph import _check_confidence_node
from app.schemas.extraction import FieldExtraction
from app.schemas.raw_extraction import (
    RawClinicalDetails,
    RawFirstAssessment,
    RawObjectiveAssessment,
    RawPatientAdvice,
)


def _high_confidence(value: str) -> FieldExtraction:
    return FieldExtraction(value=value, confidence=0.95, evidence="stated explicitly")


def _low_confidence(value: str) -> FieldExtraction:
    return FieldExtraction(value=value, confidence=0.2, evidence="guessed")


def _empty() -> FieldExtraction:
    return FieldExtraction(value="", confidence=0.0, evidence="")


def _base_raw(**overrides) -> RawFirstAssessment:
    defaults = dict(
        clinicalDetails=RawClinicalDetails(
            clinicalHistory=_empty(),
            chiefComplaint=_high_confidence("Right knee pain"),
            duration=_high_confidence("3 weeks"),
        ),
        subjectiveAssessments=[],
        objectiveAssessment=RawObjectiveAssessment(tests=[]),
        subjectiveGoals=[],
        objectiveGoals=[],
        recommendation=[],
        patientAdvice=RawPatientAdvice(adviceDetails=_empty()),
    )
    defaults.update(overrides)
    return RawFirstAssessment(**defaults)


def test_passes_when_all_present_values_are_confident():
    raw = _base_raw()
    state = _check_confidence_node({"raw": raw})
    assert state["low_confidence_fields"] == []
    assert state["assessment"] is not None
    assert state["assessment"].clinicalDetails.chiefComplaint == "Right knee pain"


def test_empty_values_do_not_trigger_low_confidence():
    # An empty value (nothing mentioned) should not be flagged - only a
    # *present* value with low confidence should be, since we never
    # hallucinate content in the first place.
    raw = _base_raw(
        clinicalDetails=RawClinicalDetails(
            clinicalHistory=_empty(),
            chiefComplaint=_high_confidence("Right knee pain"),
            duration=_empty(),
        )
    )
    state = _check_confidence_node({"raw": raw})
    assert state["low_confidence_fields"] == []
    assert state["assessment"].clinicalDetails.duration == ""


def test_low_confidence_value_fails_the_gate():
    raw = _base_raw(
        clinicalDetails=RawClinicalDetails(
            clinicalHistory=_empty(),
            chiefComplaint=_high_confidence("Right knee pain"),
            duration=_low_confidence("next month sometime"),
        )
    )
    state = _check_confidence_node({"raw": raw})
    assert state["assessment"] is None
    assert len(state["low_confidence_fields"]) == 1
    assert state["low_confidence_fields"][0].field == "clinicalDetails.duration"
