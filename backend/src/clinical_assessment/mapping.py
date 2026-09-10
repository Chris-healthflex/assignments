"""Map a verified extraction onto the production schema.

Pure and total: it drops evidence, keeps values that survived verification, and
blanks the ones that did not. Because the mapping is written out field by field
rather than derived generically, an added or renamed field fails loudly at
construction instead of quietly disappearing from the output.
"""

from .extraction_models import (
    ExtractedValue,
    RawExtraction,
    RawObjectiveGoal,
    RawObjectiveTest,
    RawRecommendation,
    RawSubjectiveAssessment,
    RawSubjectiveGoal,
)
from .grounding import GroundingReport
from .schema import (
    ClinicalDetails,
    FirstAssessment,
    ObjectiveAssessment,
    ObjectiveGoal,
    ObjectiveTest,
    PatientAdvice,
    Recommendation,
    SubjectiveAssessment,
    SubjectiveGoal,
)


def to_first_assessment(
    extraction: RawExtraction, report: GroundingReport
) -> FirstAssessment:
    """Convert a verified extraction into the assessment the frontend consumes.

    Args:
        extraction: The evidence-carrying extraction.
        report: The verdicts produced by :func:`grounding.score` for it.

    Returns:
        The production assessment, with every ungrounded field blanked.
    """
    grounded = frozenset(
        verdict.field_path for verdict in report.verdicts if verdict.grounded
    )
    details = extraction.clinicalDetails
    return FirstAssessment(
        clinicalDetails=ClinicalDetails(
            clinicalHistory=_value(
                details.clinicalHistory, "clinicalDetails.clinicalHistory", grounded
            ),
            chiefComplaint=_value(
                details.chiefComplaint, "clinicalDetails.chiefComplaint", grounded
            ),
            duration=_value(details.duration, "clinicalDetails.duration", grounded),
        ),
        subjectiveAssessments=[
            _subjective_assessment(item, f"subjectiveAssessments[{index}]", grounded)
            for index, item in enumerate(extraction.subjectiveAssessments)
        ],
        objectiveAssessment=ObjectiveAssessment(
            tests=[
                _objective_test(test, f"objectiveAssessment.tests[{index}]", grounded)
                for index, test in enumerate(extraction.objectiveAssessment.tests)
            ]
        ),
        subjectiveGoals=[
            _subjective_goal(goal, f"subjectiveGoals[{index}]", grounded)
            for index, goal in enumerate(extraction.subjectiveGoals)
        ],
        objectiveGoals=[
            _objective_goal(goal, f"objectiveGoals[{index}]", grounded)
            for index, goal in enumerate(extraction.objectiveGoals)
        ],
        recommendation=[
            _recommendation(item, f"recommendation[{index}]", grounded)
            for index, item in enumerate(extraction.recommendation)
        ],
        patientAdvice=PatientAdvice(
            adviceDetails=_value(
                extraction.patientAdvice.adviceDetails,
                "patientAdvice.adviceDetails",
                grounded,
            )
        ),
    )


def _value(extracted: ExtractedValue, field_path: str, grounded: frozenset[str]) -> str:
    """Keep a value only if its field was grounded, else blank it.

    S6 forbids emitting a clinical value we cannot support, so an unverified
    field leaves the pipeline as ``""`` rather than reaching the frontend.
    """
    return extracted.value if field_path in grounded else ""


def _subjective_assessment(
    raw: RawSubjectiveAssessment, prefix: str, grounded: frozenset[str]
) -> SubjectiveAssessment:
    """Map one subjective assessment entry."""
    return SubjectiveAssessment(
        testName=_value(raw.testName, f"{prefix}.testName", grounded),
        conclusion=_value(raw.conclusion, f"{prefix}.conclusion", grounded),
    )


def _objective_test(
    raw: RawObjectiveTest, prefix: str, grounded: frozenset[str]
) -> ObjectiveTest:
    """Map one objective test entry."""
    return ObjectiveTest(
        testName=_value(raw.testName, f"{prefix}.testName", grounded),
        unitName=_value(raw.unitName, f"{prefix}.unitName", grounded),
        value=_value(raw.value, f"{prefix}.value", grounded),
        left=_value(raw.left, f"{prefix}.left", grounded),
        right=_value(raw.right, f"{prefix}.right", grounded),
        comments=_value(raw.comments, f"{prefix}.comments", grounded),
    )


def _subjective_goal(
    raw: RawSubjectiveGoal, prefix: str, grounded: frozenset[str]
) -> SubjectiveGoal:
    """Map one subjective goal entry."""
    return SubjectiveGoal(
        goalDetails=_value(raw.goalDetails, f"{prefix}.goalDetails", grounded),
        targetDate=_value(raw.targetDate, f"{prefix}.targetDate", grounded),
    )


def _objective_goal(
    raw: RawObjectiveGoal, prefix: str, grounded: frozenset[str]
) -> ObjectiveGoal:
    """Map one objective goal entry."""
    return ObjectiveGoal(
        goalName=_value(raw.goalName, f"{prefix}.goalName", grounded),
        goalCategory=_value(raw.goalCategory, f"{prefix}.goalCategory", grounded),
        unitName=_value(raw.unitName, f"{prefix}.unitName", grounded),
        value=_value(raw.value, f"{prefix}.value", grounded),
        targetDate=_value(raw.targetDate, f"{prefix}.targetDate", grounded),
    )


def _recommendation(
    raw: RawRecommendation, prefix: str, grounded: frozenset[str]
) -> Recommendation:
    """Map one recommendation entry."""
    return Recommendation(
        sessionType=_value(raw.sessionType, f"{prefix}.sessionType", grounded),
        sessionFrequency=_value(
            raw.sessionFrequency, f"{prefix}.sessionFrequency", grounded
        ),
    )
