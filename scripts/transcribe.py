"""Transcribe a WAV to text (Whisper only) and print the transcript + metadata.

Usage:
    python scripts/transcribe.py [path/to/audio.wav]
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.transcription.whisper_service import transcribe  # noqa: E402

DEFAULT_WAV = os.path.join("data", "clinical_assessment.wav")


def main() -> int:
    wav = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_WAV
    if not os.path.exists(wav):
        print(f"WAV not found: {wav}", file=sys.stderr)
        return 2
    tr = transcribe(wav)
    print(json.dumps({"text": tr.text, **tr.as_meta()}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
