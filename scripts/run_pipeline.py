from __future__ import annotations
import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agents.extraction import run_extraction  
from app.agents.llm import ConfigurationError  
from app.config import get_settings  
from app.services.transcription import TranscriptionError, transcribe  


def _hr(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wav", type=Path, nargs="?", default=Path("data/clinical_assessment.wav"))
    parser.add_argument("--transcribe-only", action="store_true")
    parser.add_argument("--bare", action="store_true")
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    if not args.wav.is_file():
        print(f"error: no such file: {args.wav}", file=sys.stderr)
        return 2

    settings = get_settings()

    _hr(f"1. TRANSCRIBE  ({settings.whisper_backend}, model={settings.whisper_model})")
    t0 = time.perf_counter()
    try:
        transcript = transcribe(args.wav, settings)
    except TranscriptionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    took = time.perf_counter() - t0

    print(f"audio duration : {transcript.duration:.1f}s")
    print(f"language       : {transcript.language or 'unknown'}")
    print(f"segments       : {len(transcript.segments)}")
    print(f"wall time      : {took:.1f}s")
    print(f"\n{transcript.text}\n")

    if args.transcribe_only:
        return 0

    _hr(f"2. EXTRACT  (LangGraph, {settings.llm_provider}/{settings.llm_model})")
    try:
        outcome = run_extraction(transcript.text, settings)
    except ConfigurationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        print(
            "\nhint: --transcribe-only exercises the audio half of the pipeline "
            "without an LLM key.",
            file=sys.stderr,
        )
        return 1

    meta = outcome.meta.model_copy(
        update={
            "sourceFilename": args.wav.name,
            "transcriptLanguage": transcript.language,
            "audioDurationSeconds": round(transcript.duration, 2),
        }
    )

    print(f"overall confidence : {meta.overallConfidence}")
    print(f"threshold          : {settings.confidence_threshold}")
    print(f"attempts           : {meta.attempts}")
    if meta.unextractedFields:
        print("\nfields not extracted (flagged for clinician review):")
        for f in meta.unextractedFields:
            print(f"  - {f}")
    if meta.groundingWarnings:
        print("\ngrounding warnings (numbers absent from the transcript):")
        for w in meta.groundingWarnings:
            print(f"  ! {w}")
    if meta.overallConfidence < settings.confidence_threshold:
        print(
            "\nNOTE: below threshold — POST /assessments/parse would answer "
            "422 for this recording."
        )

    _hr("3. FIRSTASSESSMENT JSON")
    payload = (
        outcome.assessment.model_dump(mode="json")
        if args.bare
        else {"assessment": outcome.assessment.model_dump(mode="json"),
              "meta": meta.model_dump(mode="json")}
    )
    rendered = json.dumps(payload, indent=2, ensure_ascii=False)
    print(rendered)

    if args.out:
        args.out.write_text(rendered + "\n", encoding="utf-8")
        print(f"\nwrote {args.out}")

    if args.save:
        _hr("4. SAVE TO MONGODB")

        async def _save() -> None:
            from app.db.repository import AssessmentRepository, StorageError

            repo = AssessmentRepository(settings)
            await repo.connect()
            try:
                stored = await repo.save(outcome.assessment, meta)
                print(f"saved id: {stored.id}")
                print(f"  GET /assessments/{stored.id}")
            except StorageError as exc:
                print(f"error: {exc}", file=sys.stderr)
            finally:
                await repo.close()

        asyncio.run(_save())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
