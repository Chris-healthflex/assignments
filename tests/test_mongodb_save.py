from app.database.mongodb import MongoDB
from app.models.assessment import FirstAssessment


def main():
    assessment = FirstAssessment(
        clinicalDetails={
            "clinicalHistory": "Test clinical history",
            "chiefComplaint": "Test complaint",
            "duration": "3 weeks",
        },
        subjectiveAssessments=[],
        objectiveAssessment={"tests": []},
        subjectiveGoals=[],
        objectiveGoals=[],
        recommendation=[],
        patientAdvice={
            "adviceDetails": None
        },
    )

    database = MongoDB()

    assessment_id = database.save_assessment(assessment)

    print("Assessment saved successfully.")
    print("Assessment ID:", assessment_id)


if __name__ == "__main__":
    main()