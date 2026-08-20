"""D5 - run the full pipeline on a WAV recording and print the JSON.

    python scripts/run_pipeline.py                    # the supplied recording
    python scripts/run_pipeline.py --file other.wav
    python scripts/run_pipeline.py --raw              # bare FirstAssessment
    python scripts/run_pipeline.py --save             # also persist to MongoDB
    python scripts/run_pipeline.py --cached           # reuse a cached transcript
    python scripts/run_pipeline.py > out.json         # JSON only, report on stderr

The assessment JSON goes to stdout and the progress report to stderr, so the
output can be redirected or piped into jq without the commentary getting in
the way.

Expect roughly 25 s of transcription plus about two minutes of extraction on
local models. ``--cached`` skips transcription when iterating on extraction.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import DATA_DIR, SAMPLE_WAV, get_settings          # noqa: E402
from app.extraction.graph import extract_assessment                 # noqa: E402
from app.transcription.whisper_service import (                     # noqa: E402
    Transcript,
    WhisperTranscriber,
)


def log(message: str = "") -> None:
    """Progress goes to stderr so stdout stays pure JSON."""
    print(message, file=sys.stderr)


def rule(title: str) -> None:
    log()
    log(f"--- {title} " + "-" * max(0, 62 - len(title)))


def get_transcript(args, settings) -> Transcript:
    cache = DATA_DIR / "transcript.json"

    if args.cached and cache.exists():
        log(f"Using cached transcript: {cache.name}")
        return Transcript.model_validate_json(cache.read_text(encoding="utf-8"))

    log(f"Transcribing with {settings.whisper_backend}/{settings.whisper_model} "
        f"({settings.whisper_compute_type} on {settings.whisper_device})...")
    transcript = WhisperTranscriber(settings).transcribe(args.file)

    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(transcript.model_dump_json(indent=2), encoding="utf-8")
    return transcript


def report(transcript: Transcript, result, settings) -> None:
    confidence = result["confidence"]
    timings = result["timings"]

    rule("TRANSCRIPT")
    log(f"audio      : {transcript.durationSeconds}s, {len(transcript.segments)} segments, "
        f"language {transcript.language or 'unknown'}")
    log(f"words      : {len(transcript.text.split())}")
    log(f"text       : {transcript.text[:220]}...")

    rule("TIMINGS (seconds)")
    for stage, seconds in timings.items():
        log(f"  {stage:<18} {seconds:>7.2f}")

    rule("CONFIDENCE")
    log(f"overall    : {confidence.overall:.2f}   threshold {confidence.threshold:.2f}   "
        f"{'PASS' if confidence.meetsThreshold else 'BELOW THRESHOLD (API would return 422)'}")
    for section, sub_score in confidence.sectionScores.items():
        log(f"  {section:<24} {sub_score:.2f}")

    rule("ANTI-HALLUCINATION (S6)")
    if result["issues"]:
        log(f"{len(result['issues'])} value(s) rejected as ungrounded and cleared:")
        for issue in result["issues"]:
            log(f"  {issue.path}")
            log(f"      discarded : {issue.value[:70]!r}")
            log(f"      reason    : {issue.reason[:70]}")
    else:
        log("No values were rejected - every extracted value traced to the transcript.")

    rule("FLAGGED FIELDS (S5)")
    if confidence.flaggedFields:
        for flag in confidence.flaggedFields:
            detail = f"  {flag.detail[:50]}" if flag.detail else ""
            log(f"  [{flag.reason:<10}] {flag.path}{detail}")
    else:
        log("None.")

    if result["errors"]:
        rule("SECTION ERRORS")
        for section, message in result["errors"].items():
            log(f"  {section}: {message[:120]}")


async def persist(assessment, transcript, result, settings, filename: str) -> str:
    from app.db import client as db_client
    from app.db import repository as repo
    from app.db.models import AssessmentMetadata

    confidence = result["confidence"]
    await db_client.connect(settings)
    try:
        return await repo.save(
            assessment,
            AssessmentMetadata(
                sourceFilename=filename,
                transcript=transcript.text,
                transcriptLanguage=transcript.language,
                audioDurationSeconds=transcript.durationSeconds,
                whisperModel=transcript.model,
                whisperBackend=transcript.backend,
                llmProvider=settings.llm_provider,
                llmModel=settings.default_llm_model,
                confidence=confidence.overall,
                confidenceThreshold=confidence.threshold,
                flaggedFields=confidence.flaggedFields,
                rejectedCount=confidence.rejectedCount,
                sectionScores=confidence.sectionScores,
                timings=result["timings"],
            ),
        )
    finally:
        await db_client.disconnect()


def main() -> int:
    # Force UTF-8 on both streams. Redirected output on Windows otherwise takes
    # the locale encoding (cp1252 here), and the degree signs in every
    # range-of-motion value are written as bytes that are not valid UTF-8 - so
    # `run_pipeline.py > out.json` produced a file json.load could not read.
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        description="Run the clinical assessment pipeline on a WAV file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--file", type=Path, default=SAMPLE_WAV, help="WAV file to process")
    parser.add_argument("--raw", action="store_true", help="print the bare FirstAssessment only")
    parser.add_argument("--save", action="store_true", help="persist the result to MongoDB")
    parser.add_argument("--cached", action="store_true", help="reuse data/transcript.json if present")
    parser.add_argument("--out", type=Path, help="also write the JSON to this file")
    parser.add_argument("--quiet", action="store_true", help="suppress the progress report")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(levelname)s %(message)s",
        stream=sys.stderr,
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)

    settings = get_settings()

    if not args.file.exists():
        log(f"ERROR: no such file: {args.file}")
        return 2

    log("=" * 68)
    log("  Structured Clinical Assessment Form Filler")
    log(f"  file : {args.file.name}")
    log(f"  llm  : {settings.llm_provider}/{settings.default_llm_model}")
    log("=" * 68)

    started = time.perf_counter()

    transcript = get_transcript(args, settings)
    log(f"Transcribed in {transcript.transcribeSeconds}s. Extracting "
        f"(5 LLM calls, this is the slow part)...")

    result = extract_assessment(transcript.text, settings=settings)
    assessment = result["assessment"]
    elapsed = time.perf_counter() - started

    if not args.quiet:
        report(transcript, result, settings)

    if args.save:
        new_id = asyncio.run(persist(assessment, transcript, result, settings, args.file.name))
        rule("SAVED")
        log(f"MongoDB id: {new_id}")

    confidence = result["confidence"]
    payload = (
        assessment.model_dump()
        if args.raw
        else {
            "assessment": assessment.model_dump(),
            "transcript": {
                "text": transcript.text,
                "language": transcript.language,
                "durationSeconds": transcript.durationSeconds,
                "segments": len(transcript.segments),
                "model": transcript.model,
                "backend": transcript.backend,
            },
            "confidence": {
                "overall": confidence.overall,
                "threshold": confidence.threshold,
                "meetsThreshold": confidence.meetsThreshold,
                "sectionScores": confidence.sectionScores,
                "rejectedCount": confidence.rejectedCount,
            },
            "flaggedFields": [flag.model_dump() for flag in confidence.flaggedFields],
            "timings": result["timings"],
        }
    )

    rendered = json.dumps(payload, indent=2, ensure_ascii=False)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered, encoding="utf-8")
        log(f"\nWrote {args.out}")

    rule("ASSESSMENT JSON (stdout)")
    print(rendered)

    log(f"\nDone in {elapsed:.1f}s.")
    # Non-zero when the API would have returned 422, so CI can assert on it.
    return 0 if confidence.meetsThreshold else 1


if __name__ == "__main__":
    raise SystemExit(main())
