from app.schemas.first_assessment import FirstAssessment


def test_defaults_are_never_none():
    assessment = FirstAssessment()

    assert isinstance(assessment.subjectiveAssessments, list)
    assert isinstance(assessment.subjectiveGoals, list)
    assert isinstance(assessment.objectiveGoals, list)
    assert isinstance(assessment.recommendation, list)
    assert isinstance(assessment.objectiveAssessment.tests, list)
    assert assessment.clinicalDetails.clinicalHistory == ""
    assert assessment.patientAdvice.adviceDetails == ""


def test_single_item_arrays_stay_arrays():
    payload = {
        "subjectiveAssessments": [{"testName": "SLR", "conclusion": "Positive"}],
    }
    assessment = FirstAssessment.model_validate(payload)

    assert isinstance(assessment.subjectiveAssessments, list)
    assert len(assessment.subjectiveAssessments) == 1


def test_round_trip_matches_exact_key_names():
    assessment = FirstAssessment()
    dumped = assessment.model_dump()

    assert set(dumped.keys()) == {
        "clinicalDetails",
        "subjectiveAssessments",
        "objectiveAssessment",
        "subjectiveGoals",
        "objectiveGoals",
        "recommendation",
        "patientAdvice",
    }
