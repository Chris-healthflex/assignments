from app.models.assessment import FirstAssessment


def test_first_assessment_schema():
    data = {
        "clinicalDetails": {
            "clinicalHistory": "Patient reports knee pain.",
            "chiefComplaint": "Right knee pain",
            "duration": "3 weeks",
        },
        "subjectiveAssessments": [
            {
                "testName": "Pain",
                "conclusion": "Patient reports pain with stairs.",
            }
        ],
        "objectiveAssessment": {
            "tests": [
                {
                    "testName": "Knee flexion",
                    "unitName": "degrees",
                    "value": "120",
                    "left": "",
                    "right": "120",
                    "comments": "",
                }
            ]
        },
        "subjectiveGoals": [
            {
                "goalDetails": "Reduce knee pain.",
                "targetDate": "4 weeks",
            }
        ],
        "objectiveGoals": [
            {
                "goalName": "Knee flexion",
                "goalCategory": "ROM",
                "unitName": "degrees",
                "value": "135",
                "targetDate": "6 weeks",
            }
        ],
        "recommendation": [
            {
                "sessionType": "Physiotherapy",
                "sessionFrequency": "2 times per week",
            }
        ],
        "patientAdvice": {
            "adviceDetails": "Continue prescribed home exercises."
        },
    }

    assessment = FirstAssessment.model_validate(data)

    assert assessment.clinicalDetails.chiefComplaint == (
        "Right knee pain"
    )

    assert isinstance(
        assessment.subjectiveAssessments,
        list,
    )

    assert isinstance(
        assessment.objectiveAssessment.tests,
        list,
    )

    assert isinstance(
        assessment.subjectiveGoals,
        list,
    )

    assert isinstance(
        assessment.objectiveGoals,
        list,
    )

    assert isinstance(
        assessment.recommendation,
        list,
    )


def test_schema_rejects_extra_fields():
    data = {
        "clinicalDetails": {
            "clinicalHistory": "",
            "chiefComplaint": "",
            "duration": "",
        },
        "subjectiveAssessments": [],
        "objectiveAssessment": {
            "tests": [],
        },
        "subjectiveGoals": [],
        "objectiveGoals": [],
        "recommendation": [],
        "patientAdvice": {
            "adviceDetails": "",
        },
        "confidence": 0.95,
    }

    try:
        FirstAssessment.model_validate(data)
        assert False, "Expected validation to fail"
    except Exception:
        assert True
