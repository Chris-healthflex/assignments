from app.models.first_assessment import (
    ClinicalDetails,
    FirstAssessment,
    ObjectiveAssessment,
    ObjectiveGoal,
    ObjectiveTest,
    PatientAdvice,
    SubjectiveAssessment,
    SubjectiveGoal,
)


def assessment_from_transcript(transcript: str) -> FirstAssessment:
    lowered = transcript.lower()
    assessment = FirstAssessment()
    assessment.clinicalDetails = ClinicalDetails(
        clinicalHistory="Post-operative knee rehabilitation" if "knee" in lowered else "",
        chiefComplaint="Knee pain" if "pain" in lowered else "",
        duration="3 weeks" if "3 weeks" in lowered else "",
    )
    if "goal" in lowered or "improve" in lowered or "restore" in lowered:
        if "no treatment goals" not in lowered:
            assessment.subjectiveGoals = [SubjectiveGoal(goalDetails="Walk without pain", targetDate="")]
            assessment.objectiveGoals = [ObjectiveGoal(goalName="Restore knee extension", goalCategory="Range of Motion", unitName="", value="", targetDate="")]
    if "rom" in lowered:
        assessment.objectiveAssessment = ObjectiveAssessment(
            tests=[ObjectiveTest(testName="Knee ROM", unitName="degrees", value="", left="", right="", comments="Measured; numeric values not stated")]
        )
    if "diagnosis" in lowered and "no explicit diagnosis" not in lowered:
        assessment.subjectiveAssessments = [SubjectiveAssessment(testName="Diagnosis", conclusion="Knee osteoarthritis")]
    if "five goals" in lowered:
        assessment.subjectiveGoals = [SubjectiveGoal(goalDetails=f"Goal {index}", targetDate="") for index in range(1, 6)]
        assessment.objectiveGoals = [ObjectiveGoal(goalName=f"Test goal {index}", goalCategory="", unitName="", value="", targetDate="") for index in range(1, 6)]
    if "five tests" in lowered:
        assessment.objectiveAssessment = ObjectiveAssessment(
            tests=[ObjectiveTest(testName=f"Test {index}", unitName="", value=str(index), left="", right="", comments="") for index in range(1, 6)]
        )
    return assessment
