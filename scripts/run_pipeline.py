"""D5: run the full pipeline on a WAV file and print the FirstAssessment JSON.

Usage:
    python scripts/run_pipeline.py tests/fixtures/clinical_assessment.wav
    python scripts/run_pipeline.py <wav> --save-transcript out/transcript.txt
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.agent.graph import run_extraction  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.transcription.whisper_service import (  # noqa: E402
    TranscriptionError,
    WhisperTranscriber,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s - %(message)s")
log = logging.getLogger("pipeline")


def main() -> int:
    parser = argparse.ArgumentParser(description="WAV -> FirstAssessment JSON")
    parser.add_argument("wav", type=Path, help="Path to the .wav session recording")
    parser.add_argument("--save-transcript", type=Path, default=None)
    parser.add_argument("--save-json", type=Path, default=None)
    parser.add_argument(
        "--threshold", type=float, default=None, help="Override CONFIDENCE_THRESHOLD"
    )
    args = parser.parse_args()

    settings = get_settings()
    threshold = args.threshold if args.threshold is not None else settings.confidence_threshold

    log.info("Step 1/3  Transcribing %s with Whisper (%s)", args.wav, settings.whisper_model)
    try:
        transcript = WhisperTranscriber().transcribe(args.wav)
    except TranscriptionError as exc:
        log.error("Transcription failed: %s", exc)
        return 2
    log.info("Transcript: %d characters", len(transcript))
    if args.save_transcript:
        args.save_transcript.parent.mkdir(parents=True, exist_ok=True)
        args.save_transcript.write_text(transcript, encoding="utf-8")
        log.info("Transcript written to %s", args.save_transcript)

    log.info("Step 2/3  Running LangGraph clinical extraction")
    result = run_extraction(transcript, threshold=threshold)
    if result.error:
        log.error("Extraction failed: %s", result.error)
        return 3

    log.info("Step 3/3  Confidence gate (threshold %.2f)", threshold)
    for section, score in sorted(result.field_confidence.items()):
        marker = "ok  " if score >= threshold else "LOW "
        log.info("  %s %-22s %.2f", marker, section, score)

    payload = result.assessment.model_dump()
    rendered = json.dumps(payload, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("FirstAssessment (schema/v1)")
    print("=" * 60)
    print(rendered)

    if args.save_json:
        args.save_json.parent.mkdir(parents=True, exist_ok=True)
        args.save_json.write_text(rendered, encoding="utf-8")
        log.info("JSON written to %s", args.save_json)

    if result.low_confidence_fields:
        print("\n" + "-" * 60)
        print("Sections below threshold (the API would return HTTP 422):")
        for field in result.low_confidence_fields:
            print(f"  {field.field}: {field.confidence} < {field.threshold}")
        return 1

    print("\nAll sections met the confidence threshold.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
