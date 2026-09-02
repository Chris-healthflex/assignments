from app.models.assessment import (
    ClinicalDetails,
    FirstAssessment,
    ObjectiveAssessment,
    PatientAdvice,
)


def test_minimal_first_assessment():
    assessment = FirstAssessment(
        clinicalDetails=ClinicalDetails(
            clinicalHistory="",
            chiefComplaint="",
            duration="",
        ),
        subjectiveAssessments=[],
        objectiveAssessment=ObjectiveAssessment(
            tests=[]
        ),
        subjectiveGoals=[],
        objectiveGoals=[],
        recommendation=[],
        patientAdvice=PatientAdvice(
            adviceDetails=""
        ),
    )

    assert assessment.clinicalDetails.clinicalHistory == ""
    assert assessment.subjectiveAssessments == []
    assert assessment.objectiveAssessment.tests == []
    assert assessment.subjectiveGoals == []
    assert assessment.objectiveGoals == []
    assert assessment.recommendation == []
    assert assessment.patientAdvice.adviceDetails == ""
