"""Transcribe a WAV and cache the result under data/.

The cached transcript lets Phase 3 iterate on extraction prompts without
re-running Whisper on every attempt, which would otherwise dominate the loop.

Usage:
    python scripts/transcribe.py [--file PATH] [--model small]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import DATA_DIR, SAMPLE_WAV, get_settings           # noqa: E402
from app.transcription.whisper_service import WhisperTranscriber    # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Transcribe a WAV with Whisper.")
    parser.add_argument("--file", type=Path, default=SAMPLE_WAV)
    parser.add_argument("--model", default=None, help="Override WHISPER_MODEL")
    parser.add_argument("--out", type=Path, default=DATA_DIR / "transcript.json")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    settings = get_settings()
    if args.model:
        settings = settings.model_copy(update={"whisper_model": args.model})

    print(f"file    : {args.file.name}")
    print(f"backend : {settings.whisper_backend} / {settings.whisper_model} "
          f"({settings.whisper_compute_type} on {settings.whisper_device})")
    print("transcribing (first run also loads the model)...\n")

    transcript = WhisperTranscriber(settings).transcribe(args.file)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(transcript.model_dump_json(indent=2), encoding="utf-8")
    txt_path = args.out.with_suffix(".txt")
    txt_path.write_text(transcript.text, encoding="utf-8")

    words = len(transcript.text.split())
    print(f"language  : {transcript.language}")
    print(f"duration  : {transcript.durationSeconds}s")
    print(f"elapsed   : {transcript.transcribeSeconds}s "
          f"({transcript.durationSeconds / max(transcript.transcribeSeconds, 0.01):.1f}x realtime)")
    print(f"segments  : {len(transcript.segments)}")
    print(f"words     : {words}")
    print(f"saved     : {args.out.name}, {txt_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
