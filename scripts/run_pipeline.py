from __future__ import annotations

import json
import sys
from pathlib import Path

# Allow running:
# python scripts/run_pipeline.py
ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from app.services.pipeline import process_audio


def main():
    if len(sys.argv) > 1:
        audio_path = Path(sys.argv[1])
    else:
        audio_path = ROOT / "clinical_assessment.wav"

    if not audio_path.exists():
        print(
            f"Audio file not found: {audio_path}"
        )
        sys.exit(1)

    print(
        f"Processing: {audio_path}",
        file=sys.stderr,
    )

    try:
        assessment, transcript = process_audio(
            audio_path
        )

    except Exception as exc:
        print(
            f"Pipeline failed: {exc}",
            file=sys.stderr,
        )
        sys.exit(1)

    output = assessment.model_dump(
        mode="json"
    )

    output_directory = ROOT / "output"

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = (
        output_directory
        / "assessment.json"
    )

    output_file.write_text(
        json.dumps(
            output,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    transcript_file = (
        output_directory
        / "transcript.txt"
    )

    transcript_file.write_text(
        transcript,
        encoding="utf-8",
    )

    print(
        json.dumps(
            output,
            indent=2,
            ensure_ascii=False,
        )
    )

    print(
        f"\nSaved JSON to: {output_file}",
        file=sys.stderr,
    )

    print(
        f"Saved transcript to: {transcript_file}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
