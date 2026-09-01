from app.schemas.first_assessment import (
    FirstAssessment,
    ClinicalDetails,
    SubjectiveAssessment,
    ObjectiveAssessment,
    ObjectiveTest,
    SubjectiveGoal,
    ObjectiveGoal,
    Recommendation,
    PatientAdvice,
)
from app.core.exceptions import LowConfidenceExtractionError

# Fields we consider "critical" — if these are empty after extraction,
# we flag low confidence rather than silently saving an incomplete assessment.
CRITICAL_FIELDS = ["chiefComplaint", "clinicalHistory"]

# Fields we check for the "soft" flagging list (present but empty is OK,
# just needs to be surfaced to the reviewer, not treated as a hard failure).
OPTIONAL_TEXT_FIELDS = ["duration", "adviceDetails"]
OPTIONAL_LIST_FIELDS = [
    "subjectiveAssessments",
    "objectiveTests",
    "subjectiveGoals",
    "objectiveGoals",
    "recommendation",
]


def map_to_first_assessment(raw: dict) -> FirstAssessment:
    """
    Maps the raw dict produced by the LangGraph extraction agent into the
    strict FirstAssessment schema. Raises LowConfidenceExtractionError if
    critical fields are missing/empty — never fabricates values to fill gaps.

    Any optional field left empty by the extraction agent is recorded in
    flaggedFields, rather than being silently padded with placeholder objects.
    """
    missing = [f for f in CRITICAL_FIELDS if not raw.get(f, "").strip()]
    if missing:
        raise LowConfidenceExtractionError(missing)

    flagged_fields = []

    for field in OPTIONAL_TEXT_FIELDS:
        if not raw.get(field, "").strip():
            flagged_fields.append(field)

    for field in OPTIONAL_LIST_FIELDS:
        if not raw.get(field):
            flagged_fields.append(field)

    clinical_details = ClinicalDetails(
        clinicalHistory=raw.get("clinicalHistory", ""),
        chiefComplaint=raw.get("chiefComplaint", ""),
        duration=raw.get("duration", ""),
    )

    subjective_assessments = [
        SubjectiveAssessment(**item) for item in raw.get("subjectiveAssessments", [])
    ]

    objective_tests = [
        ObjectiveTest(**item) for item in raw.get("objectiveTests", [])
    ]
    objective_assessment = ObjectiveAssessment(tests=objective_tests)

    subjective_goals = [
        SubjectiveGoal(**item) for item in raw.get("subjectiveGoals", [])
    ]

    objective_goals = [
        ObjectiveGoal(**item) for item in raw.get("objectiveGoals", [])
    ]

    recommendation = [
        Recommendation(**item) for item in raw.get("recommendation", [])
    ]

    patient_advice = PatientAdvice(adviceDetails=raw.get("adviceDetails", ""))

    return FirstAssessment(
        clinicalDetails=clinical_details,
        subjectiveAssessments=subjective_assessments,
        objectiveAssessment=objective_assessment,
        subjectiveGoals=subjective_goals,
        objectiveGoals=objective_goals,
        recommendation=recommendation,
        patientAdvice=patient_advice,
        flaggedFields=flagged_fields,
    )