"""Unit tests for LangGraph clinical extraction agent."""

from typing import Any
from unittest.mock import MagicMock
import pytest

from app.config import settings
from app.schemas.assessment import (
    ClinicalDetails,
    FirstAssessment,
    ObjectiveAssessment,
    ObjectiveGoal,
    ObjectiveTest,
    PatientAdvice,
    Recommendation,
    SubjectiveAssessment,
    SubjectiveGoal,
)
from app.services.langgraph_agent import (
    ClinicalExtractionAgent,
    run_clinical_extraction,
)


@pytest.fixture
def representative_transcript() -> str:
    """Full representative physiotherapy clinical assessment transcript."""
    return """
    Clinician: Hello, let's review your history and current status.
    Patient: I had a road traffic accident about eight months ago. I sustained a left tibial condyle fracture and an avulsion ACL tear. I had ORIF surgery.
    Clinician: Understood. How long were you non-weight-bearing?
    Patient: Four to six weeks non-weight-bearing, then progressive loading. Right now my main complaint is left knee pain, and difficulty with walking and functional activities. Prolonged walking brings on ankle and back pain.
    Clinician: How is the pain?
    Patient: Moderate pain, mild irritability. Prolonged standing aggravates it, resting relieves it.
    Clinician: On physical exam, there is a healed medial knee surgical scar. Knee flexion is restricted and painful on the left.
    Let's record range of motion:
    - Knee flexion: left 124 degrees, right 130 degrees.
    - Knee extension: left 20 degrees, right -5 degrees.
    - Hip internal rotation: left 45 degrees, right 45 degrees.
    - Hip external rotation: left 60 degrees, right 60 degrees.
    - Ankle dorsiflexion: left 4.5 degrees, right 12 degrees.
    Patellar mobility is good.
    Our assessment is left tibial condyle fracture post-operative eight months with knee stiffness.
    We recommend physiotherapy once weekly for four sessions.
    Goals: restore left knee extension and stability, single-leg stability, quadriceps strengthening, ankle mobility.
    Advice: Avoid prolonged standing, continue progressive exercises.
    """


@pytest.fixture
def mock_extracted_assessment() -> FirstAssessment:
    """Mock FirstAssessment output matching the representative transcript."""
    return FirstAssessment(
        clinicalDetails=ClinicalDetails(
            clinicalHistory="Road traffic accident eight months ago, left tibial condyle fracture, avulsion ACL tear, ORIF, 4-6 weeks non-weight-bearing followed by progressive loading.",
            chiefComplaint="Left knee pain and difficulty with functional activities and walking.",
            duration={"text": "8 months"},
        ),
        subjectiveAssessments=[
            SubjectiveAssessment(
                testName="Pain Assessment",
                conclusion=["Moderate pain", "Mild irritability", "Aggravated by prolonged walking", "Relieved by rest"],
            )
        ],
        objectiveAssessment=ObjectiveAssessment(
            tests=[
                ObjectiveTest(
                    testName="Knee Flexion",
                    unitName="degrees",
                    value="",
                    left="124",
                    right="130",
                    comments=["Restricted and painful"],
                ),
                ObjectiveTest(
                    testName="Knee Extension",
                    unitName="degrees",
                    value="",
                    left="20",
                    right="-5",
                    comments=["Restricted extension"],
                ),
                ObjectiveTest(
                    testName="Hip Internal Rotation",
                    unitName="degrees",
                    value="",
                    left="45",
                    right="45",
                ),
                ObjectiveTest(
                    testName="Hip External Rotation",
                    unitName="degrees",
                    value="",
                    left="60",
                    right="60",
                ),
                ObjectiveTest(
                    testName="Ankle Dorsiflexion",
                    unitName="degrees",
                    value="",
                    left="4.5",
                    right="12",
                ),
            ]
        ),
        subjectiveGoals=[
            SubjectiveGoal(
                goalDetails="Walk without ankle and back pain",
                targetDate="",  # Unmentioned in transcript, remains schema-safe empty string
            )
        ],
        objectiveGoals=[
            ObjectiveGoal(
                goalName="Knee Extension",
                goalCategory="Range of Motion",
                unitName="degrees",
                value="0",
                targetDate="",  # Unmentioned, remains empty string
            )
        ],
        recommendation=[
            Recommendation(
                sessionType="Physiotherapy",
                sessionFrequency="Once weekly for 4 sessions",
            )
        ],
        patientAdvice=PatientAdvice(
            adviceDetails="Avoid prolonged standing, continue progressive loading exercises."
        ),
    )


def test_langgraph_full_workflow_execution(
    representative_transcript: str,
    mock_extracted_assessment: FirstAssessment,
):
    """Test 1 & 10: Full LangGraph workflow execution with structured extraction and validation."""
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = mock_extracted_assessment

    agent = ClinicalExtractionAgent(llm=mock_llm)
    state = agent.extract(representative_transcript)

    assert state.get("is_valid") is True
    assert len(state.get("validation_errors", [])) == 0
    assert isinstance(state.get("final_assessment"), FirstAssessment)

    final = state["final_assessment"]
    assert "left tibial condyle fracture" in final.clinicalDetails.clinicalHistory.lower()
    assert "left knee pain" in final.clinicalDetails.chiefComplaint.lower()


def test_objective_numeric_measurements_and_laterality(
    representative_transcript: str,
    mock_extracted_assessment: FirstAssessment,
):
    """Test 2, 3 & 4: Objective measurements, numeric values, units, and left/right laterality."""
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = mock_extracted_assessment

    agent = ClinicalExtractionAgent(llm=mock_llm)
    assessment, state = run_clinical_extraction(representative_transcript, agent=agent)

    tests = assessment.objectiveAssessment.tests
    assert len(tests) == 5

    # Check Knee Flexion: left 124, right 130, unit degrees
    flexion = next(t for t in tests if t.testName == "Knee Flexion")
    assert flexion.left == "124"
    assert flexion.right == "130"
    assert flexion.unitName == "degrees"

    # Check Ankle Dorsiflexion: left 4.5, right 12, unit degrees
    ankle = next(t for t in tests if t.testName == "Ankle Dorsiflexion")
    assert ankle.left == "4.5"
    assert ankle.right == "12"
    assert ankle.unitName == "degrees"


def test_recommendation_and_treatment_frequency(
    representative_transcript: str,
    mock_extracted_assessment: FirstAssessment,
):
    """Test 5: Treatment recommendation and session frequency extraction."""
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = mock_extracted_assessment

    agent = ClinicalExtractionAgent(llm=mock_llm)
    assessment, _ = run_clinical_extraction(representative_transcript, agent=agent)

    assert len(assessment.recommendation) == 1
    assert assessment.recommendation[0].sessionType == "Physiotherapy"
    assert "once weekly for 4 sessions" in assessment.recommendation[0].sessionFrequency.lower()


def test_unmentioned_target_date_remains_empty(
    representative_transcript: str,
    mock_extracted_assessment: FirstAssessment,
):
    """Test 6: Missing/unmentioned target dates remain empty string rather than invented."""
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = mock_extracted_assessment

    agent = ClinicalExtractionAgent(llm=mock_llm)
    assessment, state = run_clinical_extraction(representative_transcript, agent=agent)

    assert assessment.subjectiveGoals[0].targetDate == ""
    assert assessment.objectiveGoals[0].targetDate == ""
    assert state.get("is_valid") is True


def test_hallucinated_values_flagged_as_uncertain(representative_transcript: str):
    """Test 7 & 8: Unsupported measurements are caught and marked in uncertain_fields."""
    hallucinated_assessment = FirstAssessment(
        objectiveAssessment=ObjectiveAssessment(
            tests=[
                ObjectiveTest(
                    testName="Knee Flexion",
                    unitName="degrees",
                    left="999",  # Hallucinated value not present in transcript
                    right="130",
                )
            ]
        )
    )

    mock_llm = MagicMock()
    mock_llm.invoke.return_value = hallucinated_assessment

    agent = ClinicalExtractionAgent(llm=mock_llm)
    state = agent.extract(representative_transcript)

    assert state.get("is_valid") is False
    assert len(state.get("uncertain_fields", [])) >= 1
    assert any("999" in str(u.get("value")) for u in state.get("uncertain_fields", []))


def test_production_first_assessment_has_no_internal_metadata(
    representative_transcript: str,
    mock_extracted_assessment: FirstAssessment,
):
    """Test 9: Production FirstAssessment does not leak internal confidence, evidence, or error fields."""
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = mock_extracted_assessment

    agent = ClinicalExtractionAgent(llm=mock_llm)
    assessment, state = run_clinical_extraction(representative_transcript, agent=agent)

    dumped = assessment.model_dump()
    forbidden_keys = {"evidence", "confidence", "uncertain_fields", "validation_errors", "is_valid", "transcript"}

    # Verify no forbidden keys at top level or nested levels
    assert not any(k in dumped for k in forbidden_keys)
    assert not any(k in dumped["clinicalDetails"] for k in forbidden_keys)
    assert not any(k in dumped["objectiveAssessment"] for k in forbidden_keys)


def test_empty_transcript_handling():
    """Test handling of empty or blank transcript string."""
    agent = ClinicalExtractionAgent(llm=MagicMock())
    state = agent.extract("   ")

    assert state.get("is_valid") is False
    assert "Transcript is empty" in state.get("validation_errors", [])[0]
    assert isinstance(state.get("final_assessment"), FirstAssessment)


def test_live_openai_extraction_if_key_available(representative_transcript: str):
    """Test 13 (Optional / Live Integration): Calls live OpenAI model if OPENAI_API_KEY is configured."""
    api_key = settings.OPENAI_API_KEY.strip()
    if not api_key or api_key in {"your_openai_api_key_here", "mock_key"}:
        pytest.skip("OPENAI_API_KEY not configured for live LLM extraction test (skipping)")

    agent = ClinicalExtractionAgent()
    assessment, state = run_clinical_extraction(representative_transcript, agent=agent)

    assert isinstance(assessment, FirstAssessment)
    assert state.get("is_valid") is True
    assert len(assessment.objectiveAssessment.tests) > 0
