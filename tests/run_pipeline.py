"""
D5 — Run the full pipeline on the provided WAV and print the FirstAssessment JSON.

    python tests/run_pipeline.py                      # uses clinical_assessment.wav in repo root
    python tests/run_pipeline.py path/to/other.wav
    python tests/run_pipeline.py --session-date 2026-09-02
    python tests/run_pipeline.py --transcript-only    # skip the LLM, just print Whisper output

Exit code 0 on success, 2 if extraction confidence is below threshold
(the same condition that makes the API return 422).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.agent import run_extraction  # noqa: E402
from app.schemas import FirstAssessment  # noqa: E402
from app.transcription import transcribe  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("wav", nargs="?", default="clinical_assessment.wav")
    p.add_argument("--session-date", default=None)
    p.add_argument("--transcript-only", action="store_true")
    p.add_argument("--out", default="tests/output.json", help="where to write the JSON")
    args = p.parse_args()

    wav = Path(args.wav)
    if not wav.exists():
        print(f"WAV not found: {wav}", file=sys.stderr)
        return 1

    t0 = time.time()
    print(f"[1/3] Transcribing {wav} ...", file=sys.stderr)
    transcript = transcribe(wav.read_bytes())
    print(f"      done in {time.time()-t0:.1f}s, {len(transcript.split())} words\n", file=sys.stderr)
    print("---- TRANSCRIPT ----\n" + transcript + "\n--------------------\n", file=sys.stderr)
    if args.transcript_only:
        return 0

    t1 = time.time()
    print("[2/3] Running LangGraph extraction agent ...", file=sys.stderr)
    result = run_extraction(transcript, args.session_date)
    print(f"      done in {time.time()-t1:.1f}s\n", file=sys.stderr)

    print("[3/3] Validating against FirstAssessment schema ...", file=sys.stderr)
    payload = result.assessment.model_dump()
    FirstAssessment.model_validate(payload)  # exact-match guarantee
    assert set(payload) == set(FirstAssessment.model_fields), "schema key mismatch"

    print(json.dumps(payload, indent=2, ensure_ascii=False))
    Path(args.out).write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    print(f"\noverall_confidence = {result.overall_confidence:.2f}", file=sys.stderr)
    if result.flags:
        print("flagged fields:", file=sys.stderr)
        for f in result.flags:
            print(f"  - {f.field}  (conf {f.confidence:.2f}): {f.reason}", file=sys.stderr)
    if result.low_confidence:
        print("\nLOW CONFIDENCE — API would return 422", file=sys.stderr)
        return 2
    print(f"\nwritten to {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
