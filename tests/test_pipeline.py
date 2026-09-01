import json
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.services.transcription import transcribe_audio
from app.services.extraction import extract_clinical_data
from app.services.mapper import map_to_first_assessment
from app.core.exceptions import LowConfidenceExtractionError
from app.db.repository import save_assessment

WAV_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "clinical_assessment.wav")


def run_pipeline(wav_path: str):
    print(f"Transcribing: {wav_path}")
    transcript = transcribe_audio(wav_path)
    print("\n--- TRANSCRIPT ---")
    print(transcript)

    print("\nExtracting clinical data via LangGraph agent...")
    raw_extraction = extract_clinical_data(transcript)
    print("\n--- RAW EXTRACTION ---")
    print(json.dumps(raw_extraction, indent=2))

    print("\nMapping to FirstAssessment schema...")
    try:
        assessment = map_to_first_assessment(raw_extraction)
    except LowConfidenceExtractionError as e:
        print(f"\nLOW CONFIDENCE: missing fields {e.missing_fields}")
        return

    print("\n--- FINAL FirstAssessment JSON ---")
    print(json.dumps(assessment.model_dump(), indent=2))

    print("\nSaving to MongoDB...")
    inserted_id = save_assessment(assessment)
    print(f"Saved. Assessment ID: {inserted_id}")


if __name__ == "__main__":
    run_pipeline(WAV_PATH)