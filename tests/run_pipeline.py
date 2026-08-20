"""D5 end-to-end check: audio file -> transcript -> extraction -> MongoDB.

Usage:
    python -m tests.run_pipeline path/to/consultation.wav
    python -m tests.run_pipeline --transcript path/to/transcript.txt

Exits non-zero on the first failed stage so it can be used in CI.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from app import db
from app.config import get_settings
from app.extraction import extract
from app.schemas import ExtractionMeta, FirstAssessment
from app.transcription import transcribe


def _stage(name: str) -> None:
    print(f"\n=== {name} ===", flush=True)


async def run(audio: Path | None, transcript_file: Path | None) -> int:
    settings = get_settings()

    _stage("1. Transcription")
    if transcript_file is not None:
        transcript = transcript_file.read_text(encoding="utf-8")
        language, duration = None, None
        print(f"Loaded transcript from {transcript_file} ({len(transcript)} chars)")
    else:
        result = transcribe(audio)
        transcript, language, duration = result.text, result.language, result.duration_sec
        print(f"Transcribed {audio} -> {len(transcript)} chars, lang={language}")
    print(transcript[:400] + ("..." if len(transcript) > 400 else ""))

    _stage("2. Extraction")
    payload, warnings = extract(transcript)
    print(json.dumps(payload.model_dump(mode="json"), indent=2)[:2000])
    if warnings:
        print(f"Warnings: {warnings}")
    if payload.unresolved_fields:
        print(f"Unresolved: {payload.unresolved_fields}")

    _stage("3. Persist")
    assessment = FirstAssessment(
        patient=payload.patient,
        complaints=payload.complaints,
        medical_history=payload.medical_history,
        lifestyle=payload.lifestyle,
        goals=payload.goals,
        clinician_notes=payload.clinician_notes,
        transcript=transcript,
        meta=ExtractionMeta(
            model=settings.extraction_model,
            transcript_language=language,
            transcript_duration_sec=duration,
            unresolved_fields=payload.unresolved_fields,
            warnings=warnings,
        ),
    )
    assessment_id = await db.save_assessment(assessment)
    print(f"Saved as {assessment_id}")

    _stage("4. Read back")
    stored = await db.get_assessment(assessment_id)
    if stored is None:
        print("FAIL: could not read the assessment back", file=sys.stderr)
        return 1
    print(f"Round-tripped OK: {len(stored.complaints)} complaint(s)")

    await db.close()
    print("\nPipeline OK")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", nargs="?", type=Path, help="Audio file to transcribe")
    parser.add_argument(
        "--transcript",
        type=Path,
        help="Skip Whisper and use an existing transcript file",
    )
    args = parser.parse_args()

    if args.audio is None and args.transcript is None:
        parser.error("Provide an audio file or --transcript")

    return asyncio.run(run(args.audio, args.transcript))


if __name__ == "__main__":
    raise SystemExit(main())
