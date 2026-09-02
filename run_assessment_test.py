"""Standalone End-to-End Clinical Assessment Pipeline Runner.

Executes the complete pipeline:
WAV Audio → Whisper Transcription → LangGraph Extraction → Grounding Validation → FirstAssessment JSON.

Usage:
    python run_assessment_test.py [path_to_audio.wav]
"""

import json
from pathlib import Path
import sys
from typing import Optional

from app.config import settings
from app.schemas.assessment import FirstAssessment
from app.services.langgraph_agent import (
    ClinicalExtractionAgent,
    ExtractionState,
    run_clinical_extraction,
)
from app.services.transcriber import (
    AudioValidationError,
    TranscriptionError,
    WhisperTranscriber,
)


def run_assessment_pipeline(
    audio_path: Optional[Path] = None,
    verbose: bool = False,
) -> FirstAssessment:
    """Execute the full end-to-end clinical assessment extraction pipeline.

    Args:
        audio_path: Optional path to WAV recording (defaults to 'clinical_assessment.wav' in root).
        verbose: If True, prints step-by-step pipeline execution progress to stderr.

    Returns:
        Pure FirstAssessment instance validated against strict production schema.

    Raises:
        FileNotFoundError: If audio file is missing.
        AudioValidationError: If audio file is empty or invalid.
        TranscriptionError: If Whisper transcription fails.
        ValueError: If extraction or grounding validation fails.
    """
    target_path = audio_path or Path("clinical_assessment.wav")

    # Step 1: Validate Audio File Existence
    if not target_path.exists():
        raise FileNotFoundError(f"Audio file not found: {target_path.resolve()}")

    if verbose:
        sys.stderr.write(f"[1/4] Validating audio file: {target_path} ({target_path.stat().st_size} bytes)...\n")

    # Step 2: Whisper Audio Transcription
    transcriber = WhisperTranscriber()
    if verbose:
        sys.stderr.write(f"[2/4] Transcribing audio with Whisper ({settings.WHISPER_MODEL})...\n")

    transcript = transcriber.transcribe(target_path)
    if not transcript or not transcript.strip():
        raise TranscriptionError("Whisper returned an empty transcript from audio.")

    if verbose:
        sys.stderr.write(f"      Transcript length: {len(transcript)} characters.\n")
        sys.stderr.write(f"[3/4] Running LangGraph extraction workflow ({settings.EXTRACTION_MODEL})...\n")

    # Step 3: LangGraph Clinical Extraction & Grounding
    agent = ClinicalExtractionAgent()
    assessment, state = run_clinical_extraction(transcript, agent=agent)

    # Step 4: Validate Grounding & Anti-Hallucination Integrity
    if not state.get("is_valid", False) or state.get("validation_errors"):
        errors = state.get("validation_errors", [])
        uncertain = state.get("uncertain_fields", [])
        error_msg = f"Extraction validation failed. Errors: {errors}. Uncertain fields: {uncertain}"
        raise ValueError(error_msg)

    if verbose:
        sys.stderr.write("[4/4] Validating against strict FirstAssessment production schema...\n")

    # Final guarantee: Dump and re-validate against strict FirstAssessment
    validated_assessment = FirstAssessment.model_validate(assessment.model_dump())
    return validated_assessment


def main() -> int:
    """Main CLI entrypoint."""
    target_audio = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("clinical_assessment.wav")

    try:
        # Run pipeline with progress output on stderr so stdout contains exclusively clean JSON
        assessment = run_assessment_pipeline(audio_path=target_audio, verbose=True)

        # Output pure FirstAssessment JSON to stdout
        json_output = json.dumps(assessment.model_dump(), indent=2)
        print(json_output)
        return 0

    except FileNotFoundError as exc:
        sys.stderr.write(f"\n[ERROR: File Not Found] {str(exc)}\n")
        return 1
    except AudioValidationError as exc:
        sys.stderr.write(f"\n[ERROR: Audio Validation] {str(exc)}\n")
        return 2
    except TranscriptionError as exc:
        sys.stderr.write(f"\n[ERROR: Transcription Failure] {str(exc)}\n")
        return 3
    except ValueError as exc:
        sys.stderr.write(f"\n[ERROR: Extraction/Grounding Failure] {str(exc)}\n")
        return 4
    except Exception as exc:
        sys.stderr.write(f"\n[ERROR: Unexpected Failure] {str(exc)}\n")
        return 5


if __name__ == "__main__":
    sys.exit(main())
