"""Tests for deterministic grounding — the anti-hallucination core."""

import pytest

from clinical_assessment.extraction_models import (
    ExtractedValue,
    RawClinicalDetails,
    RawExtraction,
    RawObjectiveAssessment,
    RawObjectiveTest,
    RawPatientAdvice,
)
from clinical_assessment.grounding import (
    GroundingFailure,
    is_grounded,
    normalise,
    score,
)

TRANSCRIPT = (
    "Clinician: What brings you in? Patient: My right shoulder has been painful "
    "for about three weeks. Clinician: Let's measure it. Abduction is 120 "
    "degrees on the right, 150 on the left."
)

NOTHING_MEASURED = (
    "Clinician: What brings you in? Patient: My shoulder hurts when I reach up."
)


def blank(value: str = "", evidence: str = "") -> ExtractedValue:
    """Build an ExtractedValue, defaulting to the "not stated" pair."""
    return ExtractedValue(value=value, evidence=evidence)


def extraction_with_test(test: RawObjectiveTest) -> RawExtraction:
    """Wrap a single objective test in an otherwise blank extraction."""
    return RawExtraction(
        clinicalDetails=RawClinicalDetails(
            clinicalHistory=blank(), chiefComplaint=blank(), duration=blank()
        ),
        subjectiveAssessments=[],
        objectiveAssessment=RawObjectiveAssessment(tests=[test]),
        subjectiveGoals=[],
        objectiveGoals=[],
        recommendation=[],
        patientAdvice=RawPatientAdvice(adviceDetails=blank()),
    )


def blank_test() -> RawObjectiveTest:
    """Build an objective test with every field unset."""
    return RawObjectiveTest(
        testName=blank(),
        unitName=blank(),
        value=blank(),
        left=blank(),
        right=blank(),
        comments=blank(),
    )


def test_verbatim_quote_present_in_transcript_is_grounded() -> None:
    assert is_grounded("120", "Abduction is 120 degrees on the right", TRANSCRIPT)


@pytest.mark.parametrize(
    "evidence",
    [
        "ABDUCTION IS 120 DEGREES",
        "abduction is 120 degrees,",
        "abduction   is\n120 degrees",
        "Abduction is 120 degrees.",
    ],
)
def test_normalisation_tolerates_case_punctuation_and_whitespace(
    evidence: str,
) -> None:
    assert is_grounded("120", evidence, TRANSCRIPT)


def test_normalise_collapses_punctuation_and_whitespace() -> None:
    assert (
        normalise("  Range-of-motion:  120  DEGREES. ") == "range of motion 120 degrees"
    )


def test_invented_score_is_flagged_ungrounded() -> None:
    # The transcript states no numbers at all; the extraction claims a score.
    fabricated = blank_test()
    fabricated = fabricated.model_copy(
        update={
            "testName": blank(
                "Shoulder abduction", "My shoulder hurts when I reach up"
            ),
            "value": blank("45", "My shoulder hurts when I reach up"),
        }
    )

    report = score(extraction_with_test(fabricated), NOTHING_MEASURED)

    assert [verdict.field_path for verdict in report.ungrounded] == [
        "objectiveAssessment.tests[0].value"
    ]


def test_invented_score_fails_for_the_digit_reason_not_the_quote_reason() -> None:
    fabricated = blank_test().model_copy(
        update={"value": blank("45", "My shoulder hurts when I reach up")}
    )

    report = score(extraction_with_test(fabricated), NOTHING_MEASURED)

    assert report.ungrounded[0].failure is GroundingFailure.DIGITS_NOT_IN_EVIDENCE


def test_digit_run_must_match_whole_not_as_a_substring() -> None:
    # "45" must not be accepted on evidence that only contains "145".
    assert not is_grounded("45", "abduction is 145 degrees", "abduction is 145 degrees")


def test_quote_absent_from_transcript_is_flagged_ungrounded() -> None:
    invented_quote = blank_test().model_copy(
        update={"comments": blank("Guarded movement", "the patient guarded the joint")}
    )

    report = score(extraction_with_test(invented_quote), TRANSCRIPT)

    assert report.ungrounded[0].failure is GroundingFailure.EVIDENCE_NOT_FOUND


def test_value_without_evidence_is_flagged_ungrounded() -> None:
    unsupported = blank_test().model_copy(update={"value": blank("120", "")})

    report = score(extraction_with_test(unsupported), TRANSCRIPT)

    assert report.ungrounded[0].failure is GroundingFailure.MISSING_EVIDENCE


def test_blank_field_counts_as_grounded() -> None:
    report = score(extraction_with_test(blank_test()), TRANSCRIPT)

    assert report.ungrounded == ()


def test_confidence_of_empty_extraction_is_well_defined() -> None:
    report = score(extraction_with_test(blank_test()), "")

    assert report.confidence == 1.0


def test_confidence_is_the_grounded_ratio() -> None:
    half_wrong = blank_test().model_copy(
        update={
            "value": blank("120", "Abduction is 120 degrees"),
            "left": blank("999", "150 on the left"),
        }
    )

    report = score(extraction_with_test(half_wrong), TRANSCRIPT)

    # 3 clinicalDetails leaves + 6 in the one test + 1 patientAdvice = 10,
    # since every list but `tests` is empty. Exactly one of them fails.
    assert report.confidence == pytest.approx(9 / 10)


def test_rejected_quotes_maps_failed_paths_to_their_quotes() -> None:
    fabricated = blank_test().model_copy(
        update={"value": blank("45", "My shoulder hurts when I reach up")}
    )

    report = score(extraction_with_test(fabricated), NOTHING_MEASURED)

    assert report.rejected_quotes() == {
        "objectiveAssessment.tests[0].value": "My shoulder hurts when I reach up"
    }


def test_field_paths_index_list_entries() -> None:
    report = score(extraction_with_test(blank_test()), TRANSCRIPT)

    assert "objectiveAssessment.tests[0].testName" in {
        verdict.field_path for verdict in report.verdicts
    }
