"""Tests for the extraction to production-schema mapping."""

from test_schema import PRODUCTION_KEY_PATHS, flatten_key_paths

from clinical_assessment.extraction_models import (
    ExtractedValue,
    RawClinicalDetails,
    RawExtraction,
    RawObjectiveAssessment,
    RawObjectiveGoal,
    RawObjectiveTest,
    RawPatientAdvice,
    RawRecommendation,
    RawSubjectiveAssessment,
    RawSubjectiveGoal,
)
from clinical_assessment.grounding import score
from clinical_assessment.mapping import to_first_assessment

TRANSCRIPT = (
    "Clinician: What brings you in? Patient: My right shoulder has been painful "
    "for about three weeks, and I cannot sleep on it. Clinician: Abduction is "
    "120 degrees on the right. I would like to see you twice weekly for "
    "physiotherapy, and please apply ice after the exercises."
)


def value(text: str = "", evidence: str = "") -> ExtractedValue:
    """Build an ExtractedValue, defaulting to the "not stated" pair."""
    return ExtractedValue(value=text, evidence=evidence)


def build_extraction() -> RawExtraction:
    """Build an extraction whose every quote really is in TRANSCRIPT."""
    return RawExtraction(
        clinicalDetails=RawClinicalDetails(
            clinicalHistory=value(),
            chiefComplaint=value("Right shoulder pain", "My right shoulder has been"),
            duration=value("three weeks", "painful for about three weeks"),
        ),
        subjectiveAssessments=[
            RawSubjectiveAssessment(
                testName=value("Sleep disturbance", "I cannot sleep on it"),
                conclusion=value("Cannot sleep on the shoulder", "cannot sleep on it"),
            )
        ],
        objectiveAssessment=RawObjectiveAssessment(
            tests=[
                RawObjectiveTest(
                    testName=value("Abduction", "Abduction is 120 degrees"),
                    unitName=value("degrees", "120 degrees on the right"),
                    value=value("120", "Abduction is 120 degrees"),
                    left=value(),
                    right=value("120", "120 degrees on the right"),
                    comments=value(),
                )
            ]
        ),
        subjectiveGoals=[
            RawSubjectiveGoal(
                goalDetails=value("Sleep on the shoulder", "I cannot sleep on it"),
                targetDate=value(),
            )
        ],
        objectiveGoals=[
            RawObjectiveGoal(
                goalName=value(),
                goalCategory=value(),
                unitName=value(),
                value=value(),
                targetDate=value(),
            )
        ],
        recommendation=[
            RawRecommendation(
                sessionType=value("Physiotherapy", "twice weekly for physiotherapy"),
                sessionFrequency=value("twice weekly", "see you twice weekly"),
            )
        ],
        patientAdvice=RawPatientAdvice(
            adviceDetails=value("Apply ice after exercises", "apply ice after the")
        ),
    )


def test_mapping_output_matches_the_exact_key_set() -> None:
    extraction = build_extraction()
    report = score(extraction, TRANSCRIPT)

    dumped = to_first_assessment(extraction, report).model_dump()

    assert flatten_key_paths(dumped) == set(PRODUCTION_KEY_PATHS)


def test_grounded_values_are_carried_through_without_evidence() -> None:
    extraction = build_extraction()
    report = score(extraction, TRANSCRIPT)

    assessment = to_first_assessment(extraction, report)

    assert assessment.objectiveAssessment.tests[0].value == "120"


def test_ungrounded_field_is_blanked_not_passed_through() -> None:
    extraction = build_extraction()
    fabricated = extraction.objectiveAssessment.tests[0].model_copy(
        update={"value": value("45", "Abduction is 120 degrees")}
    )
    extraction = extraction.model_copy(
        update={"objectiveAssessment": RawObjectiveAssessment(tests=[fabricated])}
    )
    report = score(extraction, TRANSCRIPT)

    assessment = to_first_assessment(extraction, report)

    assert assessment.objectiveAssessment.tests[0].value == ""


def test_blanking_one_field_leaves_its_siblings_intact() -> None:
    extraction = build_extraction()
    fabricated = extraction.objectiveAssessment.tests[0].model_copy(
        update={"value": value("45", "Abduction is 120 degrees")}
    )
    extraction = extraction.model_copy(
        update={"objectiveAssessment": RawObjectiveAssessment(tests=[fabricated])}
    )
    report = score(extraction, TRANSCRIPT)

    assessment = to_first_assessment(extraction, report)

    assert assessment.objectiveAssessment.tests[0].right == "120"


def test_empty_section_maps_to_an_empty_array() -> None:
    extraction = build_extraction().model_copy(update={"recommendation": []})
    report = score(extraction, TRANSCRIPT)

    assessment = to_first_assessment(extraction, report)

    assert assessment.model_dump()["recommendation"] == []


def test_absent_datum_maps_to_empty_string_never_null() -> None:
    extraction = build_extraction()
    report = score(extraction, TRANSCRIPT)

    dumped = to_first_assessment(extraction, report).model_dump()

    assert dumped["clinicalDetails"]["clinicalHistory"] == ""


def test_every_leaf_of_the_output_is_a_string() -> None:
    extraction = build_extraction()
    report = score(extraction, TRANSCRIPT)

    dumped = to_first_assessment(extraction, report).model_dump()

    assert all(isinstance(leaf, str) for leaf in _leaf_values(dumped))


def _leaf_values(node: object) -> list[object]:
    """Collect every leaf value from a dumped assessment."""
    if isinstance(node, dict):
        return [leaf for child in node.values() for leaf in _leaf_values(child)]
    if isinstance(node, list):
        return [leaf for item in node for leaf in _leaf_values(item)]
    return [node]
