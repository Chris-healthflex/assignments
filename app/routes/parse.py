"""POST /assessments/parse - the audio-to-JSON pipeline endpoint."""

import logging
import os
import shutil
import tempfile

from fastapi import APIRouter, File, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse

from app.config import ALLOWED_AUDIO_EXTENSIONS
from app.schemas import ExtractionResult, FirstAssessment
from app.services.agent import run_clinical_agent
from app.services.transcription import transcribe_audio

logger = logging.getLogger(__name__)

router = APIRouter(tags=["parse"])


@router.post("/assessments/parse", response_model=FirstAssessment)
async def parse_audio_assessment(file: UploadFile = File(...)):
    """Transcribe an uploaded audio file and extract structured FirstAssessment JSON.

    Returns HTTP 422 with field-level detail if extraction confidence is below
    the threshold.
    """
    if not file.filename or not file.filename.lower().endswith(ALLOWED_AUDIO_EXTENSIONS):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid audio file format. Please upload a WAV file."
        )

    # Whisper reads from a path, so the upload is staged in a temp directory that
    # is removed in the finally block regardless of outcome.
    temp_dir = tempfile.mkdtemp()
    temp_file_path = os.path.join(temp_dir, file.filename)

    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        logger.info(f"Received audio file: {file.filename}. Starting transcription...")
        transcript = transcribe_audio(temp_file_path)
        logger.info(f"Transcription completed ({len(transcript)} chars). Running LangGraph agent...")

        extraction_result: ExtractionResult = run_clinical_agent(transcript)

        # Confidence gate. The threshold lives in app.config and is applied by
        # the agent's validate node, so this only reads the verdict.
        if not extraction_result.is_confident:
            logger.warning(
                f"Extraction confidence below threshold: {extraction_result.confidence_score}"
            )
            return JSONResponse(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                content={
                    "error": "Extraction confidence below required threshold",
                    "confidence_score": extraction_result.confidence_score,
                    "field_errors": extraction_result.field_errors,
                    "partial_assessment": extraction_result.assessment.model_dump()
                }
            )

        return extraction_result.assessment

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during audio assessment parsing: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Pipeline error: {str(e)}"
        )
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
