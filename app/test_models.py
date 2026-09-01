from models import FirstAssessment


test_data = {
    "clinicalDetails": {
        "clinicalHistory": "Patient had a road traffic accident eight months ago.",
        "chiefComplaint": "Left knee pain and difficulty walking.",
        "duration": "Eight months"
    },

    "subjectiveAssessments": [
        {
            "testName": "Pain assessment",
            "conclusion": "Moderate pain during prolonged walking and standing."
        }
    ],

    "objectiveAssessment": {
        "tests": [
            {
                "testName": "Knee flexion",
                "unitName": "degrees",
                "value": "124",
                "left": "124",
                "right": "130",
                "comments": "Left knee flexion restricted."
            }
        ]
    },

    "subjectiveGoals": [
        {
            "goalDetails": "Return to full functional activity.",
            "targetDate": "Not specified"
        }
    ],

    "objectiveGoals": [
        {
            "goalName": "Improve knee extension",
            "goalCategory": "Range of motion",
            "unitName": "degrees",
            "value": "Improve",
            "targetDate": "Not specified"
        }
    ],

    "recommendation": [
        {
            "sessionType": "Physiotherapy",
            "sessionFrequency": "Once weekly for four sessions"
        }
    ],

    "patientAdvice": {
        "adviceDetails": "Improve knee extension, stability, strength and ankle mobility."
    }
}


assessment = FirstAssessment(**test_data)

print("Schema validation successful!")
print(assessment.model_dump_json(indent=4))