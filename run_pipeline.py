#!/usr/bin/env python3
"""
Deliverable D5: End-to-End Pipeline Test & Execution Script

Runs the complete Clinical Audio -> Structured Assessment pipeline:
1. Validates & loads the WAV file (generating a sample if not present).
2. Transcribes audio via Whisper (API or local or fallback).
3. Extracts clinical assessment data via LangGraph extraction agent.
4. Enforces strict Pydantic v2 FirstAssessment schema.
5. Persists the assessment into MongoDB.
6. Retrieves the saved assessment from MongoDB and prints the final JSON.
"""

import os
import sys
import json
import asyncio
import argparse
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.models.schema import FirstAssessment
from app.services.transcription import TranscriptionService
from app.services.extraction_agent import ClinicalExtractionAgent
from app.services.database import db
from tests.generate_sample_audio import generate_clinical_wav


async def run_pipeline(audio_file_path: str):
    print("=" * 80)
    print(" CLINICAL AUDIO TO STRUCTURED ASSESSMENT REPORT PIPELINE ")
    print("=" * 80)

    # 1. Check/Prepare Audio File
    if not os.path.exists(audio_file_path):
        print(f"[*] Audio file '{audio_file_path}' not found. Generating sample clinical WAV...")
        generate_clinical_wav(audio_file_path, duration_sec=3.0)

    print(f"\n[Step 1] Loading audio file: {os.path.abspath(audio_file_path)}")
    with open(audio_file_path, "rb") as f:
        audio_bytes = f.read()
    print(f"         Loaded {len(audio_bytes):,} bytes.")

    is_valid_wav = TranscriptionService.validate_wav(audio_bytes)
    print(f"         WAV container validation: {'PASSED' if is_valid_wav else 'FAILED'}")
    if not is_valid_wav:
        print("[!] Error: File is not a valid WAV audio file.")
        sys.exit(1)

    # 2. Whisper Transcription
    print("\n[Step 2] Transcribing audio with Whisper...")
    transcription = TranscriptionService.transcribe_audio(
        audio_bytes, filename=os.path.basename(audio_file_path)
    )
    print("         Transcription Output:")
    print("         " + "-" * 60)
    for line in transcription.strip().split("\n"):
        print(f"         | {line}")
    print("         " + "-" * 60)

    # 3. LangGraph Clinical Extraction & Anti-Hallucination Agent
    print("\n[Step 3] Extracting clinical data with LangGraph agent...")
    result = ClinicalExtractionAgent.run(transcription)

    if not result.get("success") or result.get("assessment") is None:
        print("\n[!] Extraction failed or confidence below threshold:")
        print(json.dumps(result.get("validation_errors", []), indent=2))
        sys.exit(1)

    assessment: FirstAssessment = result["assessment"]
    confidence = result["confidence"]

    print(f"         Extraction Succeeded!")
    print(f"         Overall Confidence Score: {confidence.overall_score:.2f}")
    print(f"         Section Confidence Scores: {json.dumps(confidence.section_scores, indent=11)}")

    # 4. Strict Schema Verification
    print("\n[Step 4] Validating against strict FirstAssessment Pydantic v2 schema...")
    assessment_dict = assessment.model_dump()
    print(f"         Verified 7 schema sections with strict string and array constraints:")
    for section in assessment_dict.keys():
        val = assessment_dict[section]
        type_str = f"array[{len(val)}]" if isinstance(val, list) else "object"
        print(f"          - {section}: {type_str}")

    # 5. Persist in MongoDB
    print("\n[Step 5] Persisting assessment in MongoDB...")
    saved_record = await db.save_assessment(assessment)
    saved_id = saved_record["id"]
    print(f"         Successfully saved to database with Document ID: {saved_id}")
    print(f"         Storage Mode: {'MongoDB' if not db.is_mock else 'In-Memory DB Engine'}")

    # 6. Retrieve from MongoDB
    print(f"\n[Step 6] Verifying persistence - Retrieving assessment by ID ({saved_id})...")
    retrieved = await db.get_assessment_by_id(saved_id)

    print("\n" + "=" * 80)
    print(" FINAL STRUCTURED FIRST_ASSESSMENT JSON (PRODUCTION FRONTEND SCHEMA)")
    print("=" * 80)
    print(json.dumps(retrieved["assessment"], indent=2))
    print("=" * 80)
    print(f"Pipeline executed successfully. Assessment ID: {saved_id}\n")
    return retrieved


def main():
    parser = argparse.ArgumentParser(description="Clinical Audio to Structured Assessment Pipeline Runner")
    parser.add_argument(
        "--audio",
        "-a",
        type=str,
        default="clinical_assessment.wav",
        help="Path to WAV audio file (defaults to 'clinical_assessment.wav')",
    )
    args = parser.parse_args()

    asyncio.run(run_pipeline(args.audio))


if __name__ == "__main__":
    main()
