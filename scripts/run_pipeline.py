"""D5 — run the full pipeline on a WAV and print the resulting JSON.

Usage:
    python scripts/run_pipeline.py [path/to/audio.wav] [--save]
    python scripts/run_pipeline.py --transcript "raw text ..."   # skip Whisper

Defaults to data/clinical_assessment.wav. Set USE_STUB_LLM=1 to run without Ollama.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.services.pipeline import run_from_transcript, run_from_wav  # noqa: E402

DEFAULT_WAV = os.path.join("data", "clinical_assessment.wav")


def main() -> int:
    ap = argparse.ArgumentParser(description="Run the clinical assessment pipeline.")
    ap.add_argument("wav", nargs="?", default=DEFAULT_WAV, help="Path to a WAV file")
    ap.add_argument("--transcript", help="Skip Whisper; extract from this text")
    ap.add_argument("--save", action="store_true", help="Persist result to the DB")
    args = ap.parse_args()

    if args.transcript:
        result = run_from_transcript(args.transcript)
    else:
        if not os.path.exists(args.wav):
            print(f"WAV not found: {args.wav}", file=sys.stderr)
            return 2
        result = run_from_wav(args.wav)

    payload = result.model_dump(mode="json")

    if args.save:
        from app.db.client import db
        from app.db import repository

        async def _save():
            await db.connect()
            new_id = await repository.save(payload)
            await db.close()
            return new_id

        payload["id"] = asyncio.run(_save())

    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
