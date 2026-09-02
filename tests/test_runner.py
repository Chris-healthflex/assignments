"""Unit tests for standalone run_assessment_test.py pipeline runner."""

from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from app.schemas.assessment import FirstAssessment
from run_assessment_test import run_assessment_pipeline


@pytest.fixture
def dummy_wav(tmp_path: Path) -> Path:
    """Create dummy WAV file for runner testing."""
    wav = tmp_path / "clinical_test.wav"
    wav.write_bytes(b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x44\xac\x00\x00data\x00\x00\x00\x00")
    return wav


def test_runner_pipeline_mock_execution(dummy_wav: Path):
    """Verify that run_assessment_pipeline executes end-to-end and returns strict FirstAssessment."""
    sample_transcript = "Patient presents with left knee pain following a road traffic accident."

    with patch("run_assessment_test.WhisperTranscriber") as mock_transcriber_cls:
        mock_transcriber = MagicMock()
        mock_transcriber.transcribe.return_value = sample_transcript
        mock_transcriber_cls.return_value = mock_transcriber

        with patch("run_assessment_test.ClinicalExtractionAgent") as mock_agent_cls:
            mock_agent = MagicMock()
            expected_assessment = FirstAssessment()
            mock_agent.extract.return_value = {
                "transcript": sample_transcript,
                "raw_assessment": expected_assessment,
                "evidence": {},
                "uncertain_fields": [],
                "validation_errors": [],
                "final_assessment": expected_assessment,
                "is_valid": True,
            }
            mock_agent_cls.return_value = mock_agent

            result = run_assessment_pipeline(audio_path=dummy_wav, verbose=False)

            assert isinstance(result, FirstAssessment)
            dumped = result.model_dump()
            assert "clinicalDetails" in dumped
            assert "objectiveAssessment" in dumped
            assert "id" not in dumped


def test_runner_missing_file_raises_error():
    """Verify missing audio file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        run_assessment_pipeline(audio_path=Path("non_existent_recording.wav"))
