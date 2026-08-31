import pytest

from app.errors import PipelineError
from app.extraction import AssessmentDraft, extract_assessment
from tests.conftest import TRANSCRIPT, StubLLM


async def test_the_workflow_maps_the_transcript_into_the_schema(settings):
    result = await extract_assessment(TRANSCRIPT, settings, StubLLM())

    assessment = result.assessment.model_dump()
    assert assessment["clinicalDetails"]["chiefComplaint"] == (
        "Left knee pain and difficulty walking"
    )
    assert assessment["objectiveAssessment"]["tests"][0] == {
        "testName": "Knee flexion",
        "unitName": "degrees",
        "value": "",
        "left": "124",
        "right": "130",
        "comments": "",
    }
    assert assessment["recommendation"][0]["sessionType"] == "Physiotherapy"
    assert result.confidence == pytest.approx(0.9)


async def test_fields_the_transcript_does_not_state_are_tracked_not_invented(settings):
    result = await extract_assessment(TRANSCRIPT, settings, StubLLM())

    assessment = result.assessment.model_dump()
    assert assessment["patientAdvice"]["adviceDetails"] == ""
    assert assessment["objectiveGoals"] == []
    assert "patientAdvice.adviceDetails" in result.unextracted_fields
    assert "objectiveGoals" in result.unextracted_fields
    assert "subjectiveGoals[0].targetDate" in result.unextracted_fields
    assert "clinicalDetails.chiefComplaint" not in result.unextracted_fields


async def test_low_confidence_raises_422_with_field_details(settings):
    llm = StubLLM(
        confidence=0.2,
        unsupported=["objectiveTests[0].right"],
        notes="the measurements are not stated",
    )

    with pytest.raises(PipelineError) as excinfo:
        await extract_assessment(TRANSCRIPT, settings, llm)

    error = excinfo.value
    assert error.status_code == 422
    assert error.code == "low_extraction_confidence"
    assert error.details[0] == {
        "field": "confidence",
        "message": "the measurements are not stated",
        "value": 0.2,
        "threshold": 0.6,
    }
    assert error.details[1]["field"] == "objectiveTests[0].right"


async def test_empty_draft_entries_are_dropped(settings):
    draft = AssessmentDraft.model_validate(
        {
            "clinicalDetails": {"chiefComplaint": "Left knee pain"},
            "objectiveTests": [{"testName": None}],
            "recommendation": [{"sessionType": "  ", "sessionFrequency": None}],
        }
    )

    result = await extract_assessment(TRANSCRIPT, settings, StubLLM(draft=draft))

    assert result.assessment.objectiveAssessment.tests == []
    assert result.assessment.recommendation == []


async def test_a_model_failure_is_reported_as_a_pipeline_error(settings):
    llm = StubLLM()
    llm.draft = RuntimeError("rate limited")

    with pytest.raises(PipelineError) as excinfo:
        await extract_assessment(TRANSCRIPT, settings, llm)

    assert excinfo.value.code == "extraction_failed"
    assert "rate limited" in excinfo.value.details[0]["message"]


async def test_a_transcript_that_is_too_short_is_rejected(settings):
    with pytest.raises(PipelineError) as excinfo:
        await extract_assessment("hello", settings, StubLLM())

    assert excinfo.value.status_code == 422
    assert excinfo.value.details[0]["field"] == "transcript"
