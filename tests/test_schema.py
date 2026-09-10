from app.models.schemas import FirstAssessment


def test_first_assessment_schema():
    assessment = FirstAssessment(
        clinicalDetails={
            "clinicalHistory": "History of left knee injury following road traffic accident",
            "chiefComplaint": "Left knee pain and difficulty walking",
            "duration": {
                "value": "8",
                "unit": "months",
            },
        },
        subjectiveAssessments=[
            {
                "testName": "Pain assessment",
                "conclusion": [
                    "Moderate pain during prolonged walking and standing"
                ],
            }
        ],
        objectiveAssessment={
            "tests": [
                {
                    "testName": "Knee flexion",
                    "unitName": "degrees",
                    "value": "124",
                    "left": "124",
                    "right": "130",
                    "comments": [
                        "Left knee flexion is restricted compared with the right"
                    ],
                }
            ]
        },
        subjectiveGoals=[
            {
                "goalDetails": "Return to full functional activity",
                "targetDate": ["Not specified"],
            }
        ],
        objectiveGoals=[
            {
                "goalName": "Knee extension",
                "goalCategory": "Range of motion",
                "unitName": "degrees",
                "value": "5",
                "targetDate": ["Not specified"],
            }
        ],
        recommendation=[
            {
                "sessionType": "Physiotherapy",
                "sessionFrequency": "Once weekly for four sessions",
            }
        ],
        patientAdvice={
            "adviceDetails": {
                "details": "Continue prescribed physiotherapy exercises"
            }
        },
    )

    assert assessment.clinicalDetails.chiefComplaint
    assert isinstance(assessment.subjectiveAssessments, list)
    assert isinstance(assessment.objectiveAssessment.tests, list)
    assert isinstance(assessment.subjectiveGoals, list)
    assert isinstance(assessment.objectiveGoals, list)
    assert isinstance(assessment.recommendation, list)
    assert isinstance(assessment.patientAdvice.adviceDetails, dict)