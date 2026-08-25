"""Unit tests for the graph and the transcription guard rails."""
from __future__ import annotations

import pytest

from app.agent.graph import run_extraction
from app.models.assessment import SECTION_ALIASES, FirstAssessment
from app.models.internal import ExtractionEnvelope
from app.transcription.whisper_service import TranscriptionError, validate_wav
from tests.conftest import FakeLLM, full_confidence, make_assessment


def test_graph_produces_valid_assessment():
    result = run_extraction("Patient reports knee pain.", llm=FakeLLM())
    assert result.error is None
    assert result.passed
    assert isinstance(result.assessment, FirstAssessment)


def test_graph_flags_low_confidence_fields():
    scores = full_confidence()
    scores["patientAdvice"] = 0.1
    llm = FakeLLM(ExtractionEnvelope(assessment=make_assessment(), field_confidence=scores))
    result = run_extraction("mumbled audio", llm=llm)
    assert not result.passed
    assert [f.field for f in result.low_confidence_fields] == ["patientAdvice"]


def test_missing_confidence_score_counts_as_failure():
    llm = FakeLLM(
        ExtractionEnvelope(assessment=make_assessment(), field_confidence={})
    )
    result = run_extraction("transcript", llm=llm)
    assert len(result.low_confidence_fields) == len(SECTION_ALIASES)


def test_empty_transcript_short_circuits():
    result = run_extraction("   ", llm=FakeLLM())
    assert result.error is not None
    assert result.assessment is None


def test_llm_failure_is_captured():
    result = run_extraction("transcript", llm=FakeLLM(raises=True))
    assert result.error is not None
    assert "extraction failed" in result.error.lower()


def test_threshold_is_configurable():
    scores = full_confidence(0.5)
    llm = FakeLLM(ExtractionEnvelope(assessment=make_assessment(), field_confidence=scores))
    assert run_extraction("t", llm=llm, threshold=0.4).passed
    assert not run_extraction("t", llm=llm, threshold=0.9).passed


def test_validate_wav_rejects_missing_file(tmp_path):
    with pytest.raises(TranscriptionError, match="not found"):
        validate_wav(tmp_path / "nope.wav")


def test_validate_wav_rejects_wrong_extension(tmp_path):
    bad = tmp_path / "audio.mp3"
    bad.write_bytes(b"data")
    with pytest.raises(TranscriptionError, match="Expected a .wav"):
        validate_wav(bad)


def test_validate_wav_rejects_non_wav_content(tmp_path):
    fake = tmp_path / "audio.wav"
    fake.write_bytes(b"this is not a RIFF container")
    with pytest.raises(TranscriptionError):
        validate_wav(fake)


def test_validate_wav_accepts_real_wav(tmp_path, wav_bytes):
    good = tmp_path / "good.wav"
    good.write_bytes(wav_bytes)
    validate_wav(good)
