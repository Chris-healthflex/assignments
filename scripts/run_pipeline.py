"""Run the whole pipeline on a WAV file and print the FirstAssessment JSON.

Usage:
    python scripts/run_pipeline.py [path/to/recording.wav]
"""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings
from app.errors import PipelineError
from app.extraction import extract_assessment
from app.transcription import transcribe

DEFAULT_AUDIO = Path("clinical_assessment.wav")


async def main() -> int:
    # Transcripts contain characters such as the degree sign, which a default
    # Windows console (cp1252) cannot print.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    audio = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_AUDIO
    if not audio.is_file():
        print(f"audio file not found: {audio}", file=sys.stderr)
        return 1

    settings = get_settings()
    print(f"audio     : {audio}")
    print(f"whisper   : {settings.whisper_backend} ({settings.whisper_model})")
    print(f"extraction: {settings.llm_model}")

    try:
        transcript = await transcribe(audio, settings)
        print(f"\n--- transcript ---\n{transcript}")

        result = await extract_assessment(transcript, settings)
    except PipelineError as exc:
        print(f"\npipeline failed [{exc.code}]: {exc.message}", file=sys.stderr)
        for detail in exc.details:
            print(f"  {detail}", file=sys.stderr)
        return 1

    print(f"\nconfidence  : {result.confidence:.2f}")
    print(f"unextracted : {result.unextracted_fields or 'none'}")
    if result.unsupported_fields:
        print(f"unsupported : {result.unsupported_fields}")

    print("\n--- FirstAssessment ---")
    print(json.dumps(result.assessment.model_dump(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
