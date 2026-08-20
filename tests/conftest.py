import io
import math
import struct
import wave
from typing import Generator
import mongomock
import pytest
from fastapi.testclient import TestClient

from app.core.config import get_settings
from app.database.mongodb import db_manager
from app.repositories.assessment_repository import AssessmentRepository, get_assessment_repository
from app.schemas.first_assessment import (
    FirstAssessment,
    ClinicalDetails,
    SubjectiveAssessment,
    ObjectiveAssessment,
    ObjectiveTest,
    SubjectiveGoal,
    ObjectiveGoal,
    Recommendation,
    PatientAdvice,
)
from app.services.audio_validator import AudioValidator, get_audio_validator
from app.services.transcription import MockWhisperTranscriber, get_transcriber
from app.services.extraction import ClinicalExtractionService, get_extraction_service
from app.services.assessment_service import AssessmentService, get_assessment_service
from app.main import app


def create_mock_wav_bytes(duration_sec: float = 1.0, sample_rate: int = 16000) -> bytes:
    """Generates a valid binary WAV file stream in memory."""
    buf = io.BytesIO()
    num_samples = int(sample_rate * duration_sec)

    with wave.open(buf, "wb") as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)

        # Generate simple sine wave tone
        frames = bytearray()
        for i in range(num_samples):
            value = int(32767.0 * 0.3 * math.sin(2.0 * math.pi * 440.0 * i / sample_rate))
            frames.extend(struct.pack("<h", value))

        wav_file.writeframes(frames)

    return buf.getvalue()


@pytest.fixture
def mock_mongo_client() -> Generator[mongomock.MongoClient, None, None]:
    """Provides an in-memory mongomock client for database operations."""
    client = mongomock.MongoClient()
    db_manager.set_client_override(client)
    yield client
    db_manager.close()


@pytest.fixture
def sample_wav_content() -> bytes:
    """Returns valid WAV bytes for testing."""
    return create_mock_wav_bytes(duration_sec=1.5, sample_rate=16000)


@pytest.fixture
def sample_first_assessment() -> FirstAssessment:
    """Returns a fully populated valid FirstAssessment instance."""
    return FirstAssessment(
        clinicalDetails=ClinicalDetails(
            clinicalHistory="History of mild left knee meniscus strain",
            chiefComplaint="Knee pain after running",
            duration="3 weeks",
        ),
        subjectiveAssessments=[
            SubjectiveAssessment(
                testName="Pain Scale (VAS)",
                conclusion="Patient rates pain 6/10 during running",
            )
        ],
        objectiveAssessment=ObjectiveAssessment(
            tests=[
                ObjectiveTest(
                    testName="Knee Flexion",
                    unitName="degrees",
                    value="125",
                    left="120",
                    right="135",
                    comments="Left knee slightly restricted compared to right",
                )
            ]
        ),
        subjectiveGoals=[
            SubjectiveGoal(
                goalDetails="Return to running 5km without discomfort",
                targetDate="2026-10-15",
            )
        ],
        objectiveGoals=[
            ObjectiveGoal(
                goalName="Left Knee Flexion",
                goalCategory="Range of Motion",
                unitName="degrees",
                value="135",
                targetDate="2026-10-15",
            )
        ],
        recommendation=[
            Recommendation(
                sessionType="Physiotherapy",
                sessionFrequency="Twice per week",
            )
        ],
        patientAdvice=PatientAdvice(
            adviceDetails="Apply cold compress for 15 minutes twice daily after walking."
        ),
    )


@pytest.fixture
def test_client(mock_mongo_client: mongomock.MongoClient) -> Generator[TestClient, None, None]:
    """Provides a TestClient with mock dependencies injected."""
    # Configure mock transcriber
    mock_transcriber = MockWhisperTranscriber(
        mock_transcript=(
            "Patient presents with knee pain for 3 weeks. "
            "Knee flexion was measured at 125 degrees. "
            "Goal is returning to 5k run by October 15. "
            "We advise physiotherapy twice per week and icing daily."
        )
    )
    repo = AssessmentRepository(collection=mock_mongo_client[get_settings().MONGODB_DATABASE]["assessments"])
    extraction_service = ClinicalExtractionService()
    assessment_service = AssessmentService(
        audio_validator=AudioValidator(),
        transcriber=mock_transcriber,
        extraction_service=extraction_service,
        repository=repo,
    )

    app.dependency_overrides[get_assessment_service] = lambda: assessment_service
    app.dependency_overrides[get_assessment_repository] = lambda: repo
    app.dependency_overrides[get_transcriber] = lambda: mock_transcriber

    with TestClient(app) as client:
        yield client

    app.dependency_overrides.clear()
