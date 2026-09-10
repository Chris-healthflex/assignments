from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app
from app.services.transcription import transcribe_audio
from app.services.extraction import (
    AssessmentExtractionError,
    extract_assessment,
)


AUDIO_FILE = Path("clinical_assessment.wav")


def test_extraction_pipeline():
    assert AUDIO_FILE.exists(), "clinical_assessment.wav is missing"

    print("\nTranscribing audio locally...")
    transcript = transcribe_audio(AUDIO_FILE)

    assert transcript.strip()

    print("\nExtracting clinical assessment...")
    assessment = extract_assessment(transcript)

    assert assessment is not None

    print("\n--- STRUCTURED ASSESSMENT ---")
    print(assessment.model_dump_json(indent=2))
    print("--- END STRUCTURED ASSESSMENT ---")

    # Confirm the result follows the required top-level structure.
    data = assessment.model_dump()

    assert set(data.keys()) == {
        "clinicalDetails",
        "subjectiveAssessments",
        "objectiveAssessment",
        "subjectiveGoals",
        "objectiveGoals",
        "recommendation",
        "patientAdvice",
    }

    assert isinstance(data["subjectiveAssessments"], list)
    assert isinstance(data["objectiveAssessment"]["tests"], list)
    assert isinstance(data["subjectiveGoals"], list)
    assert isinstance(data["objectiveGoals"], list)
    assert isinstance(data["recommendation"], list)

def test_parse_returns_422_when_extraction_confidence_is_low(monkeypatch):
    def fake_transcribe_audio(_path):
        return "The patient reports some discomfort."

    def fake_extract_assessment(_transcript):
        raise AssessmentExtractionError(
            missing_fields=[
                "clinicalDetails.clinicalHistory",
                "clinicalDetails.chiefComplaint",
                "clinicalDetails.duration",
            ],
            confidence=0.8,
        )

    monkeypatch.setattr(
        "app.api.assessments.transcribe_audio",
        fake_transcribe_audio,
    )

    monkeypatch.setattr(
        "app.api.assessments.extract_assessment",
        fake_extract_assessment,
    )

    client = TestClient(app)

    response = client.post(
        "/assessments/parse",
        files={
            "file": (
                "test.wav",
                b"fake wav content",
                "audio/wav",
            )
        },
    )

    assert response.status_code == 422

    data = response.json()

    assert data["detail"]["confidence"] == 0.8

    assert data["detail"]["missing_fields"] == [
        "clinicalDetails.clinicalHistory",
        "clinicalDetails.chiefComplaint",
        "clinicalDetails.duration",
    ]