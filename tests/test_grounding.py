from app.models.first_assessment import ClinicalDetails, FirstAssessment, ObjectiveAssessment, ObjectiveTest
from app.pipeline.grounding import ground_assessment, verify_value


TRANSCRIPT = (
    "The patient reports left knee pain. The injury occurred eight months ago. "
    "Objective measurements showed left knee flexion of 124 degrees compared "
    "with 130 degrees on the right. Physiotherapy was recommended once weekly."
)


def test_real_values_are_grounded() -> None:
    assert verify_value("124", TRANSCRIPT)[0]
    assert verify_value("8 months", TRANSCRIPT)[0]
    assert verify_value("degrees", TRANSCRIPT)[0]
    assert verify_value("left knee pain", TRANSCRIPT)[0]
    assert verify_value("Strength", "The plan includes strengthening the quadriceps.")[0]


def test_invented_number_is_rejected() -> None:
    grounded, reason = verify_value("140", TRANSCRIPT)
    assert not grounded
    assert "140" in reason


def test_invented_date_is_rejected() -> None:
    grounded, reason = verify_value("2026-09-01", TRANSCRIPT)
    assert not grounded
    assert "Date" in reason


def test_invented_clinical_text_is_rejected() -> None:
    grounded, reason = verify_value("rheumatoid arthritis and diabetes", TRANSCRIPT)
    assert not grounded
    assert "meaningful words" in reason


def test_grounding_clears_unsupported_nested_values_and_preserves_shape() -> None:
    assessment = FirstAssessment(
        clinicalDetails=ClinicalDetails(
            chiefComplaint="left knee pain",
            clinicalHistory="prior stroke in 2019",
            duration="eight months",
        ),
        objectiveAssessment=ObjectiveAssessment(
            tests=[ObjectiveTest(testName="Knee flexion", left="140", right="130")]
        ),
    )
    grounded, issues = ground_assessment(assessment, TRANSCRIPT)
    assert grounded.clinicalDetails.chiefComplaint == "left knee pain"
    assert grounded.clinicalDetails.clinicalHistory == ""
    assert grounded.objectiveAssessment.tests[0].left == ""
    assert grounded.objectiveAssessment.tests[0].right == "130"
    assert {issue["field"] for issue in issues} == {
        "clinicalDetails.clinicalHistory",
        "objectiveAssessment.tests[0].left",
    }
