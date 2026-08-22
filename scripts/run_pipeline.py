import os
import sys
import json
import logging
from dotenv import load_dotenv

# Load env variables from local .env
load_dotenv()

# Add project root to python path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.transcription import transcribe_audio
from app.graph.assessment_graph import app_graph
from app.models.assessment import FirstAssessment

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run_pipeline")

def main():
    # 1. Locate clinical_assessment.wav at the root directory
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    wav_path = os.path.join(project_root, "clinical_assessment.wav")
    
    logger.info(f"Looking for audio file at: {wav_path}")
    if not os.path.exists(wav_path):
        logger.error(f"Audio file clinical_assessment.wav not found at {wav_path}!")
        sys.exit(1)
        
    # 2. Run Whisper Transcription
    logger.info("Transcribing audio file...")
    try:
        transcript = transcribe_audio(wav_path)
        logger.info("Transcription completed successfully.")
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        sys.exit(2)
        
    # 3. Run the LangGraph Extraction pipeline
    logger.info("Running LangGraph workflow...")
    try:
        state = app_graph.invoke({"transcript": transcript})
    except Exception as e:
        logger.error(f"LangGraph execution failed: {e}")
        sys.exit(3)
        
    # 4. Check for validation errors
    errors = state.get("validation_errors") or []
    if errors:
        logger.error("Pipeline failed confidence validation!")
        logger.error(json.dumps(errors, indent=2))
        sys.exit(4)
        
    # 5. Output the final JSON
    assessment_dict = state.get("first_assessment")
    if not assessment_dict:
        logger.error("Pipeline did not yield any assessment.")
        sys.exit(5)
        
    try:
        # Validate against the strict FirstAssessment schema
        FirstAssessment.model_validate(assessment_dict)
        print("\n=== FINAL VALIDATED ASSESSMENT JSON ===")
        print(json.dumps(assessment_dict, indent=2))
        print("=======================================\n")
        logger.info("Pipeline executed and validated successfully.")
    except Exception as e:
        logger.error(f"Final output failed validation against Pydantic schema: {e}")
        sys.exit(6)

if __name__ == "__main__":
    main()
