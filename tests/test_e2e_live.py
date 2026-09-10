import os
from pathlib import Path
import pytest

from app.services.audio import validate_wav_header, load_and_resample_wav
from app.services.transcription import transcribe
from app.services.extraction import build_pipeline_graph
from app.models.assessment import FirstAssessment


@pytest.mark.slow
def test_audio_and_transcription_live():
    """Validates the audio decoding and Whisper transcription on the real clinical_assessment.wav."""
    audio_path = Path("clinical_assessment.wav")
    assert audio_path.exists(), "clinical_assessment.wav must exist in repository root."

    # 1. Validate header without ffmpeg
    with open(audio_path, "rb") as f:
        validate_wav_header(f)

    # 2. Resample without ffmpeg
    samples = load_and_resample_wav(audio_path, target_sr=16000)
    assert len(samples) > 0
    duration = len(samples) / 16000
    assert 100 < duration < 115  # ~105s

    # 3. Transcribe with Whisper
    result = transcribe(audio_path, model_size="base")
    assert len(result.text) > 200
    assert "knee" in result.text.lower()
    assert len(result.segments) > 10


@pytest.mark.live
def test_full_pipeline_live_with_llm():
    """Runs the full pipeline if an LLM key is configured."""
    has_key = any(
        os.environ.get(k)
        for k in ["GEMINI_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"]
    )
    if not has_key:
        pytest.skip("No LLM API key found in environment (GEMINI_API_KEY / OPENAI_API_KEY).")

    audio_path = Path("clinical_assessment.wav")
    transcript_result = transcribe(audio_path, model_size="base")

    pipeline = build_pipeline_graph()
    res = pipeline.invoke({
        "transcript": transcript_result.text,
        "segments": transcript_result.segments,
    })

    assert "assessment" in res
    assessment: FirstAssessment = res["assessment"]
    assert assessment.clinicalDetails.chiefComplaint != ""
    assert len(assessment.objectiveAssessment.tests) > 0

    assert "confidence_report" in res
    report = res["confidence_report"]
    assert 0.0 <= report.overall <= 1.0
