#!/usr/bin/env python
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.agent import run_extraction_pipeline
from app.db import save_assessment
from app.transcription import transcribe_audio


def main():
    parser = argparse.ArgumentParser(
        description="Run end-to-end clinical assessment pipeline on a WAV audio file."
    )
    parser.add_argument("audio_path", nargs="?", default="clinical_assessment.wav", help="Path to WAV file")
    parser.add_argument("--session-date", default=None, help="Session date (YYYY-MM-DD) for relative temporal resolution")
    parser.add_argument("--transcript-only", action="store_true", help="Only run transcription and exit")
    parser.add_argument("--save", action="store_true", help="Persist output assessment to MongoDB")
    parser.add_argument("--output", default="output_assessment.json", help="Path to output assessment JSON file")

    args = parser.parse_args()

    audio_file = Path(args.audio_path)
    if not audio_file.exists():
        sys.stderr.write(f"Error: Audio file '{args.audio_path}' not found.\n")
        sys.exit(1)

    sys.stderr.write(f"Transcribing '{args.audio_path}' with Whisper...\n")
    try:
        transcript = transcribe_audio(str(audio_file))
    except Exception as e:
        sys.stderr.write(f"Transcription failed: {e}\n")
        sys.exit(1)

    # Save transcript
    with open("transcript.txt", "w", encoding="utf-8") as f:
        f.write(transcript)
    sys.stderr.write(f"Transcript saved to transcript.txt ({len(transcript)} chars)\n")

    if args.transcript_only:
        print(transcript)
        sys.exit(0)

    sys.stderr.write("Running LangGraph clinical extraction & anti-hallucination audit...\n")
    try:
        result = run_extraction_pipeline(transcript, session_date=args.session_date)
    except Exception as e:
        sys.stderr.write(f"Extraction pipeline failed: {e}\n")
        sys.exit(1)

    output_dict = result.assessment.model_dump()
    json_str = json.dumps(output_dict, indent=2)

    # Save structured JSON
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(json_str)
    sys.stderr.write(f"Assessment JSON written to {args.output}\n")

    # Output to stdout
    print(json_str)

    # Summary
    sys.stderr.write("\n" + "=" * 50 + "\n")
    sys.stderr.write(f"Extraction Overall Confidence: {result.overall_confidence}\n")
    if result.flags:
        sys.stderr.write(f"Flags detected ({len(result.flags)}):\n")
        for flag in result.flags:
            sys.stderr.write(f"  - [{flag.field}] conf={flag.confidence}: {flag.reason}\n")
    else:
        sys.stderr.write("All fields verified with high confidence (0 flags).\n")
    sys.stderr.write("=" * 50 + "\n")

    # Optional Mongo Save
    if args.save:
        sys.stderr.write("Saving to MongoDB...\n")
        record = asyncio.run(
            save_assessment(
                result.assessment,
                meta={
                    "source_file": str(audio_file),
                    "confidence": result.overall_confidence,
                    "flags": [f.model_dump() for f in result.flags],
                }
            )
        )
        sys.stderr.write(f"Saved to MongoDB with document ID: {record.id}\n")

    if result.low_confidence:
        sys.stderr.write("Notice: Extraction confidence below threshold (exit code 2).\n")
        sys.exit(2)

    sys.exit(0)


if __name__ == "__main__":
    main()
