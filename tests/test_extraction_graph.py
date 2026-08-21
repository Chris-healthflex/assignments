from app.schemas.first_assessment import (
    ClinicalDetails,
    FirstAssessment,
    SubjectiveAssessment,
)
from app.services.extraction_graph import ExtractionResult, run_extraction


class FakeLLM:
    """Stands in for ChatOpenAI(...).with_structured_output(...) in tests."""

    def __init__(self, result: ExtractionResult):
        self._result = result
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        return self._result


def test_run_extraction_returns_assessment_and_confidence_flag():
    fake_result = ExtractionResult(
        assessment=FirstAssessment(
            clinicalDetails=ClinicalDetails(chiefComplaint="Lower back pain"),
            subjectiveAssessments=[
                SubjectiveAssessment(testName="SLR", conclusion="Positive")
            ],
        ),
        low_confidence_sections=[],
    )
    llm = FakeLLM(fake_result)

    result, is_low_confidence = run_extraction("some transcript", llm=llm)

    assert llm.calls == 1
    assert result.assessment.clinicalDetails.chiefComplaint == "Lower back pain"
    assert is_low_confidence is False


def test_run_extraction_flags_low_confidence_past_threshold():
    fake_result = ExtractionResult(
        assessment=FirstAssessment(),
        low_confidence_sections=["subjectiveGoals", "objectiveGoals"],
    )
    llm = FakeLLM(fake_result)

    _, is_low_confidence = run_extraction(
        "sparse transcript", llm=llm, confidence_threshold=2
    )

    assert is_low_confidence is True


def test_run_extraction_below_threshold_is_not_flagged():
    fake_result = ExtractionResult(
        assessment=FirstAssessment(),
        low_confidence_sections=["subjectiveGoals"],
    )
    llm = FakeLLM(fake_result)

    _, is_low_confidence = run_extraction(
        "mostly complete transcript", llm=llm, confidence_threshold=2
    )

    assert is_low_confidence is False


def test_run_extraction_drops_unknown_section_names():
    fake_result = ExtractionResult(
        assessment=FirstAssessment(),
        low_confidence_sections=["not_a_real_section"],
    )
    llm = FakeLLM(fake_result)

    result, _ = run_extraction("transcript", llm=llm)

    assert result.low_confidence_sections == []
