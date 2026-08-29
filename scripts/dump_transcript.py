"""
Usage:
    python scripts/dump_transcript.py clinical_assessment.wav
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.transcription import transcribe_wav  # noqa: E402


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python scripts/dump_transcript.py <path_to_wav_file>")
        sys.exit(1)

    wav_path = Path(sys.argv[1])
    if not wav_path.exists():
        print(f"File not found: {wav_path}")
        sys.exit(1)

    print(f"Transcribing {wav_path} ...")
    transcript = transcribe_wav(wav_path.read_bytes(), wav_path.name)

    out_path = wav_path.with_suffix(".transcript.txt")
    out_path.write_text(transcript, encoding="utf-8")

    print("\n--- TRANSCRIPT ---\n")
    print(transcript)
    print(f"\n--- Saved to {out_path} ---")


if __name__ == "__main__":
    main()
