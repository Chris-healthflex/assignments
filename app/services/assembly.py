from app.schemas.extraction import ExtractionResult, ExtractionField
from app.schemas.assessment import (
    ClinicalDetails, SubjectiveAssessment, ObjectiveAssessment, ObjectiveTest,
    SubjectiveGoal, ObjectiveGoal, Recommendation, PatientAdvice, FirstAssessment
)
from app.guardrails.source_match import adjust_confidence

def _field_to_str(field: ExtractionField) -> str:
    if not field.is_mentioned or field.value is None:
        return ""
    return str(field.value).strip()

def assemble_first_assessment(result: ExtractionResult, transcript: str) -> FirstAssessment:
    # Clinical details
    cd = result.clinicalDetails
    clinical_details = ClinicalDetails(
        clinicalHistory=_field_to_str(cd.clinicalHistory),
        chiefComplaint=_field_to_str(cd.chiefComplaint),
        duration=_field_to_str(cd.duration)
    )

    # Subjective assessments
    subjective_assessments = [
        SubjectiveAssessment(
            testName=_field_to_str(item.testName),
            conclusion=_field_to_str(item.conclusion)
        )
        for item in result.subjectiveAssessments
    ]

    # Objective assessment
    objective_tests = [
        ObjectiveTest(
            testName=_field_to_str(t.testName),
            unitName=_field_to_str(t.unitName),
            value=_field_to_str(t.value),
            left=_field_to_str(t.left),
            right=_field_to_str(t.right),
            comments=_field_to_str(t.comments)
        )
        for t in result.objectiveAssessment.tests
    ]
    objective_assessment = ObjectiveAssessment(tests=objective_tests)

    # Subjective goals
    subjective_goals = [
        SubjectiveGoal(
            goalDetails=_field_to_str(g.goalDetails),
            targetDate=_field_to_str(g.targetDate)
        )
        for g in result.subjectiveGoals
    ]

    # Objective goals
    objective_goals = [
        ObjectiveGoal(
            goalName=_field_to_str(g.goalName),
            goalCategory=_field_to_str(g.goalCategory),
            unitName=_field_to_str(g.unitName),
            value=_field_to_str(g.value),
            targetDate=_field_to_str(g.targetDate)
        )
        for g in result.objectiveGoals
    ]

    # Recommendations
    recommendations = [
        Recommendation(
            sessionType=_field_to_str(r.sessionType),
            sessionFrequency=_field_to_str(r.sessionFrequency)
        )
        for r in result.recommendation
    ]

    # Patient advice
    pa = result.patientAdvice
    patient_advice = PatientAdvice(adviceDetails=_field_to_str(pa.adviceDetails))

    return FirstAssessment(
        clinicalDetails=clinical_details,
        subjectiveAssessments=subjective_assessments,
        objectiveAssessment=objective_assessment,
        subjectiveGoals=subjective_goals,
        objectiveGoals=objective_goals,
        recommendation=recommendations,
        patientAdvice=patient_advice
    )