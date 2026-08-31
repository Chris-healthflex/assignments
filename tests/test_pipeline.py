"""
D5 — Test script: run pipeline on the provided WAV, print JSON.

Usage:
    python -m tests.test_pipeline path/to/clinical_assessment.wav

This runs transcription + extraction directly (bypassing the HTTP layer)
so it can be used standalone to sanity-check the pipeline, and also
imports cleanly under pytest for the two schema-level assertions below.
"""
from __future__ import annotations

import json
import sys

from app.models.schema import FirstAssessment
from app.services.extraction_agent import run_extraction
from app.services.transcription import transcribe_wav


def run(wav_path: str) -> dict:
    print(f"[1/3] Transcribing {wav_path} with Whisper...", file=sys.stderr)
    transcription = transcribe_wav(wav_path)
    print(f"    -> {len(transcription.text)} chars, language={transcription.language}", file=sys.stderr)

    print("[2/3] Running LangGraph extraction agent...", file=sys.stderr)
    result = run_extraction(transcription.text)

    print("[3/3] Validating against FirstAssessment schema...", file=sys.stderr)
    FirstAssessment.model_validate(result.assessment.model_dump())  # raises if non-conformant

    output = {
        "assessment": result.assessment.model_dump(),
        "overall_confidence": result.overall_confidence,
        "low_confidence_fields": result.low_confidence_fields,
        "transcript_preview": result.transcript[:500],
    }
    return output


# --- pytest-compatible smoke tests (schema-level, no audio/LLM required) ---

def test_empty_assessment_matches_schema():
    """An all-default FirstAssessment must still be schema-valid: every
    array present as [], every string present as ''."""
    empty = FirstAssessment()
    dumped = empty.model_dump()
    assert dumped["subjectiveAssessments"] == []
    assert dumped["objectiveGoals"] == []
    assert dumped["clinicalDetails"]["chiefComplaint"] == ""
    assert isinstance(dumped["objectiveAssessment"]["tests"], list)


def test_schema_rejects_extra_fields():
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        FirstAssessment.model_validate({"clinicalDetails": {}, "unexpectedField": 1})


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python -m tests.test_pipeline path/to/clinical_assessment.wav")
        sys.exit(1)

    output = run(sys.argv[1])
    print(json.dumps(output, indent=2))
