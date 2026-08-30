import pytest

from app.config import Settings
from app.errors import ExtractionConfidenceError
from app.pipeline.extraction_agent import ExtractionResult, extract_assessment
from app.schemas.first_assessment import FirstAssessment


def settings() -> Settings:
    return Settings(confidence_threshold=0.6)


@pytest.mark.asyncio
async def test_empty_transcript():
    with pytest.raises(ExtractionConfidenceError) as exc:
        await extract_assessment("", settings(), extractor=lambda _: {})

    assert exc.value.low_confidence_fields == ["transcript"]


@pytest.mark.asyncio
async def test_low_confidence_extraction():
    async def extractor(_: str):
        return ExtractionResult(
            assessment=FirstAssessment(),
            confidence={"clinicalDetails.chiefComplaint": 0.3},
        )

    with pytest.raises(ExtractionConfidenceError) as exc:
        await extract_assessment("patient has knee pain", settings(), extractor=extractor)

    assert exc.value.low_confidence_fields == ["clinicalDetails.chiefComplaint"]


@pytest.mark.asyncio
async def test_successful_extraction_path():
    async def extractor(_: str):
        return {
            "assessment": {
                "clinicalDetails": {
                    "clinicalHistory": None,
                    "chiefComplaint": "right knee pain",
                    "duration": "two weeks",
                },
                "subjectiveAssessments": [],
                "objectiveAssessment": {"tests": []},
                "subjectiveGoals": [],
                "objectiveGoals": [],
                "recommendation": [],
                "patientAdvice": {"adviceDetails": ""},
            },
            "confidence": {"clinicalDetails": 0.9},
        }

    assessment, confidence = await extract_assessment(
        "patient reports right knee pain for two weeks", settings(), extractor=extractor
    )

    assert assessment.clinicalDetails.chiefComplaint == "right knee pain"
    assert assessment.clinicalDetails.clinicalHistory == ""
    assert confidence == {"clinicalDetails": 0.9}
