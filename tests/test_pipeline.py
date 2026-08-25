"""Unit/API checks and a runnable full-pipeline script for the supplied WAV."""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

# Make the project root importable when this script is run directly,
# e.g. `python tests/test_pipeline.py <audio.wav>` (running a script by path
# only adds the script's own directory to sys.path, not the project root).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient

from app.main import app
from app.schemas import ExtractionEnvelope, FirstAssessment


def example_assessment() -> FirstAssessment:
    return FirstAssessment.model_validate({
        "clinicalDetails": {"clinicalHistory": "", "chiefComplaint": "knee pain", "duration": "two weeks"},
        "subjectiveAssessments": [], "objectiveAssessment": {"tests": []},
        "subjectiveGoals": [], "objectiveGoals": [], "recommendation": [],
        "patientAdvice": {"adviceDetails": ""},
    })


def test_schema_rejects_unknown_keys():
    payload = example_assessment().model_dump()
    payload["unexpected"] = "no"
    try:
        FirstAssessment.model_validate(payload)
    except Exception:
        return
    raise AssertionError("The schema must reject extra response fields")


def test_parse_returns_field_level_422(monkeypatch):
    from app import main

    monkeypatch.setattr(main, "transcribe_wav", lambda _: "brief transcript")
    monkeypatch.setattr(
        main, "extract_assessment",
        lambda _: ExtractionEnvelope.model_validate({
            "assessment": example_assessment().model_dump(),
            "uncertain_fields": [{"field": "clinicalDetails.duration", "reason": "not stated"}],
        }),
    )
    response = TestClient(app).post("/assessments/parse", files={"file": ("test.wav", b"RIFF", "audio/wav")})
    assert response.status_code == 422
    assert response.json()["detail"][0]["field"] == "clinicalDetails.duration"


def main() -> None:
    from app.agent import extract_assessment
    from app.transcribe import transcribe_wav

    wav_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(r"C:\Users\akjee\Downloads\clinical_assessment.wav")
    transcript = transcribe_wav(wav_path)
    result = extract_assessment(transcript)
    if result.uncertain_fields:
        print(json.dumps({"detail": [issue.model_dump() for issue in result.uncertain_fields]}, indent=2))
        raise SystemExit(2)
    print(result.assessment.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
