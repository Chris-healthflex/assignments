from app.models.first_assessment import FirstAssessment


def test_first_assessment_has_exact_top_level_sections() -> None:
    assessment = FirstAssessment()
    assert set(assessment.model_dump()) == {
        "clinicalDetails",
        "subjectiveAssessments",
        "objectiveAssessment",
        "subjectiveGoals",
        "objectiveGoals",
        "recommendation",
        "patientAdvice",
    }
    assert assessment.subjectiveAssessments == []
    assert assessment.objectiveAssessment.tests == []


def test_unknown_schema_fields_are_rejected() -> None:
    try:
        FirstAssessment(unexpected="value")
    except Exception as exc:
        assert "unexpected" in str(exc)
    else:
        raise AssertionError("Unknown fields must be rejected")
