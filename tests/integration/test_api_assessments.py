def test_create_and_get_assessment(client):
    assessment_data = {
        "clinicalDetails": {"clinicalHistory": "", "chiefComplaint": "", "duration": ""},
        "subjectiveAssessments": [],
        "objectiveAssessment": {"tests": []},
        "subjectiveGoals": [],
        "objectiveGoals": [],
        "recommendation": [],
        "patientAdvice": {"adviceDetails": ""}
    }
    resp = client.post("/api/v1/assessments", json=assessment_data)
    assert resp.status_code == 201
    aid = resp.json()["id"]
    resp2 = client.get(f"/api/v1/assessments/{aid}")
    assert resp2.status_code == 200
    assert resp2.json()["assessment"] == assessment_data