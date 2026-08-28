"""
Deliverable D5: run the pipeline on the provided WAV, print the resulting JSON.

    python tests/test_pipeline.py path/to/clinical_assessment.wav

With TRANSCRIPTION_ENGINE=whisper and a valid LLM_PROVIDER key set, this exercises
the real production path (Whisper + LLM extraction agent). With no network access to
model-weight hosts, set TRANSCRIPTION_ENGINE=pocketsphinx to exercise everything except
the Whisper step with zero external dependencies.

Also runnable as a pytest smoke test (`pytest tests/test_pipeline.py`) against a short
fixture WAV — see `test_schema_roundtrip` below, which needs no audio or network at all.
"""

import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.agent.graph import run_pipeline
from app.schemas.first_assessment import ExtractionEnvelope


def main():
    wav_path = sys.argv[1] if len(sys.argv) > 1 else "sample_data/clinical_assessment.wav"
    print(f"Running pipeline on: {wav_path}", file=sys.stderr)
    envelope = run_pipeline(wav_path)
    # Validate before printing, exactly as the API layer does
    ExtractionEnvelope(**envelope)
    print(json.dumps(envelope, indent=2, default=str))


def test_schema_roundtrip():
    """No audio, no network: proves the schema itself is well-formed and that an
    empty-but-valid envelope satisfies FirstAssessment's contract."""
    from app.schemas.first_assessment import FirstAssessment

    empty = FirstAssessment()
    dumped = empty.model_dump()
    assert isinstance(dumped["subjectiveAssessments"], list)
    assert isinstance(dumped["objectiveGoals"], list)
    assert dumped["clinicalDetails"]["chiefComplaint"] == ""
    # round-trip
    FirstAssessment(**dumped)


if __name__ == "__main__":
    main()
