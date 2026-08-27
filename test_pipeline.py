import sys
import json
import asyncio
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Windows consoles default to cp1252, which cannot encode characters that appear
# in Whisper transcripts (smart quotes, accents). Force UTF-8 so a printing error
# can never mask a successful pipeline run.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from app.config import CONFIDENCE_THRESHOLD
from app.services.agent import run_clinical_agent
from app.services.storage import (
    get_assessment_by_id,
    list_assessments,
    save_assessment_to_db,
)
from app.services.transcription import transcribe_audio

async def main():
    wav_file = "clinical_assessment.wav"
    if len(sys.argv) > 1:
        wav_file = sys.argv[1]

    path = Path(wav_file)
    if not path.exists():
        print(f"Error: Target audio file '{wav_file}' not found.")
        sys.exit(1)

    print("=" * 70)
    print(f"STANCE HEALTH CLINICAL ASSESSMENT PIPELINE TEST")
    print(f"Target Audio File: {wav_file} ({path.stat().st_size / (1024*1024):.2f} MB)")
    print("=" * 70)

    # 1. Transcribe Audio
    # The try guards only the transcription call. Wrapping the prints too would
    # let an unrelated output error be reported as a transcription failure.
    print("\n[Step 1/3] Transcribing audio with Whisper...")
    try:
        transcript = transcribe_audio(str(path))
    except Exception as e:
        print(f"[FAIL] Transcription failed: {e}")
        sys.exit(1)

    print(f"[OK] Transcription successful ({len(transcript)} chars)")
    print(f"\n--- Transcript Snippet ---\n{transcript[:300]}...\n--------------------------\n")

    # 2. Extract Clinical Entities via LangGraph Agent
    print("[Step 2/3] Extracting structured FirstAssessment via LangGraph Agent...")
    extraction_result = run_clinical_agent(transcript)

    print(f"Confidence Score: {extraction_result.confidence_score:.2f}")
    print(f"Is Confident: {extraction_result.is_confident}")
    if extraction_result.field_errors:
        print(f"Field Errors/Warnings: {extraction_result.field_errors}")

    # The extracted JSON is the deliverable, so print it unconditionally -
    # including when the confidence gate rejects it.
    assessment_json = extraction_result.assessment.model_dump()
    print("\n================== FIRST ASSESSMENT JSON OUTPUT ==================")
    print(json.dumps(assessment_json, indent=2))
    print("==================================================================\n")

    if not extraction_result.is_confident:
        print(f"Pipeline flagged low confidence (below {CONFIDENCE_THRESHOLD} threshold).")
        print("POST /assessments/parse would return HTTP 422 with:")
        err_json = {
            "status_code": 422,
            "error": "Extraction confidence below required threshold",
            "confidence_score": extraction_result.confidence_score,
            "field_errors": extraction_result.field_errors
        }
        print(json.dumps(err_json, indent=2))
        print("\nSkipping database steps: low-confidence extractions are not persisted.")
        return

    print("[OK] Extraction conforms to FirstAssessment schema.")

    # 3. Test Database Persistence
    print("\n[Step 3/3] Testing database save & retrieval...")
    saved_doc = await save_assessment_to_db(assessment_json)
    doc_id = saved_doc["id"]
    print(f"[OK] Saved to database with ID: {doc_id}")

    fetched_doc = await get_assessment_by_id(doc_id)
    if fetched_doc:
        print(f"[OK] Retrieved document from DB successfully (ID: {fetched_doc['id']})")

    all_docs = await list_assessments()
    print(f"[OK] Total assessments in database: {len(all_docs)}")

    print("\nPipeline test completed successfully.")

if __name__ == "__main__":
    asyncio.run(main())
