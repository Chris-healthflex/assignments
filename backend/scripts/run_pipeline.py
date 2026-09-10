"""Transcribe a session recording and print its assessment as JSON.

Progress goes to stderr and the assessment to stdout, so the output can be
piped straight into a file or a JSON tool:

    python scripts/run_pipeline.py clinical_assessment.wav > assessment.json

On PowerShell, ``>`` prepends a UTF-8 BOM; pipe to
``Out-File -Encoding utf8NoBOM`` instead.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import anthropic

from clinical_assessment.agent import AssessmentOutcome, run_agent
from clinical_assessment.errors import ClinicalAssessmentError
from clinical_assessment.llm import request_extraction
from clinical_assessment.transcription import transcribe


def main(argv: list[str] | None = None) -> int:
    """Run the pipeline over one recording.

    Args:
        argv: Command-line arguments; defaults to ``sys.argv[1:]``.

    Returns:
        0 on success, 1 on any pipeline or input failure.
    """
    arguments = _parse_arguments(argv)
    logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(message)s")

    if not arguments.wav_path.is_file():
        print(f"error: {arguments.wav_path} does not exist", file=sys.stderr)
        return 1

    try:
        outcome = _extract_assessment(arguments.wav_path)
    except ClinicalAssessmentError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    _report_confidence(outcome)
    json.dump(outcome.assessment.model_dump(), sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


def _parse_arguments(argv: list[str] | None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "wav_path", type=Path, help="16-bit PCM WAV recording of the session"
    )
    return parser.parse_args(argv)


def _extract_assessment(wav_path: Path) -> AssessmentOutcome:
    """Transcribe the recording, then extract and verify an assessment."""
    print(f"Transcribing {wav_path.name}...", file=sys.stderr)
    transcript = transcribe(wav_path)
    print(f"Transcript: {len(transcript)} characters", file=sys.stderr)

    client = anthropic.Anthropic()
    return run_agent(transcript, lambda prompt: request_extraction(client, prompt))


def _report_confidence(outcome: AssessmentOutcome) -> None:
    """Write confidence data to stderr, keeping stdout pure JSON."""
    print(
        f"Confidence {outcome.confidence:.2f} after {outcome.attempts} repair round(s)",
        file=sys.stderr,
    )
    for verdict in outcome.ungrounded_fields:
        reason = verdict.failure.value if verdict.failure else "unverified"
        print(f"  unverified: {verdict.field_path} - {reason}", file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main())
