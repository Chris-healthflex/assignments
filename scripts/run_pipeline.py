from __future__ import annotations

import json
import sys
from pathlib import Path

# Make `app` importable when run as `python scripts/run_pipeline.py`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

from app.services.assessment_service import ConfidenceTooLowError, parse_wav_to_assessment  # noqa: E402


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python scripts/run_pipeline.py <path_to_wav_file>")
        sys.exit(1)

    wav_path = Path(sys.argv[1])
    if not wav_path.exists():
        print(f"File not found: {wav_path}")
        sys.exit(1)

    print(f"Reading {wav_path} ...")
    file_bytes = wav_path.read_bytes()

    print("Transcribing with Whisper...")
    print("Running LangGraph clinical extraction agent...")
    print("Validating against FirstAssessment schema + confidence gate...")

    try:
        assessment = parse_wav_to_assessment(file_bytes, wav_path.name)
    except ConfidenceTooLowError as exc:
        print("\n422 Unprocessable Entity — confidence too low for these fields:")
        print(json.dumps(exc.low_confidence_fields, indent=2))
        sys.exit(2)

    print("\nFirstAssessment JSON:\n")
    print(json.dumps(assessment.model_dump(), indent=2))


if __name__ == "__main__":
    main()
