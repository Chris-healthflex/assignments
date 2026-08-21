#!/usr/bin/env python3
"""Run the full Whisper -> LangGraph pipeline on a WAV file and print the
resulting FirstAssessment JSON.

Usage:
    python scripts/run_pipeline.py path/to/clinical_assessment.wav
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from groq import Groq  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.services.extraction_graph import run_extraction  # noqa: E402
from app.services.transcription import TranscriptionError, transcribe_audio  # noqa: E402


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python scripts/run_pipeline.py <path-to-wav>", file=sys.stderr)
        return 1

    wav_path = Path(sys.argv[1])
    if not wav_path.exists():
        print(f"File not found: {wav_path}", file=sys.stderr)
        return 1

    settings = get_settings()
    if not settings.groq_api_key:
        print(
            "GROQ_API_KEY is not set. Copy .env.example to .env and fill it in.",
            file=sys.stderr,
        )
        return 1

    try:
        transcript = transcribe_audio(wav_path, client=Groq(api_key=settings.groq_api_key))
    except TranscriptionError as exc:
        print(f"Transcription failed: {exc}", file=sys.stderr)
        return 1

    print("--- Transcript ---", file=sys.stderr)
    print(transcript, file=sys.stderr)

    result, is_low_confidence = run_extraction(
        transcript,
        confidence_threshold=settings.confidence_flag_threshold,
        api_key=settings.groq_api_key,
    )

    print("\n--- FirstAssessment JSON ---", file=sys.stderr)
    print(json.dumps(result.assessment.model_dump(), indent=2))

    if is_low_confidence:
        print(
            f"\nWARNING: low-confidence sections: {result.low_confidence_sections}",
            file=sys.stderr,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
