import logging
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, UploadFile, File, HTTPException, Query, status
from fastapi.responses import JSONResponse

from app.config import settings
from app.models.schema import (
    FirstAssessment,
    AssessmentSaveResponse,
    AssessmentListResponse,
)
from app.services.transcription import TranscriptionService
from app.services.extraction_agent import ClinicalExtractionAgent
from app.services.database import db

router = APIRouter(prefix="/assessments", tags=["Assessments"])
logger = logging.getLogger(__name__)


@router.post(
    "/parse",
    response_model=FirstAssessment,
    summary="EP1 — Parse Audio Session to FirstAssessment",
    status_code=status.HTTP_200_OK,
    responses={
        200: {
            "description": "Structured FirstAssessment JSON successfully extracted",
            "model": FirstAssessment,
        },
        422: {
            "description": "Extraction confidence below threshold or clinical parsing unprocessable",
        },
        400: {
            "description": "Invalid file format. WAV format is required.",
        },
    },
)
async def parse_audio(file: UploadFile = File(...)):
    """
    Step 1: Accept WAV file upload.
    Step 2: Transcribe audio using Whisper.
    Step 3: Extract clinical data using LangGraph/LangChain agent.
    Step 4: Map strictly into FirstAssessment schema.
    Step 5: Enforce confidence threshold; return HTTP 422 on low confidence or validation errors.
    """
    filename = file.filename or "audio.wav"
    if not filename.lower().endswith(".wav") and file.content_type not in ["audio/wav", "audio/x-wav", "audio/wave"]:
        # Log warning but continue if bytes are valid WAV
        logger.warning(f"File {filename} may not be WAV MIME type: {file.content_type}")

    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "Uploaded file is empty."},
        )

    # 1. Validate WAV format
    if not TranscriptionService.validate_wav(audio_bytes):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "Invalid WAV file structure. Please upload a valid uncompressed PCM WAV file."},
        )

    # 2. Transcribe audio with Whisper
    try:
        transcription = TranscriptionService.transcribe_audio(audio_bytes, filename=filename)
    except Exception as e:
        logger.error(f"Whisper transcription error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": f"Audio transcription service failed: {str(e)}"},
        )

    if not transcription or len(transcription.strip()) == 0:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "error": "Transcription returned empty audio content.",
                "confidence_score": 0.0,
                "field_errors": [{"loc": ["audio"], "msg": "No speech detected in audio.", "type": "empty_transcription"}]
            },
        )

    # 3 & 4. LangGraph Clinical Extraction & Confidence Evaluation
    result = ClinicalExtractionAgent.run(transcription)

    # 5. Check confidence & validation
    if not result.get("success") or result.get("assessment") is None:
        confidence = result.get("confidence")
        overall_score = confidence.overall_score if confidence else 0.0
        flagged = confidence.flagged_fields if confidence else []
        notes = confidence.notes if confidence else []
        val_errors = result.get("validation_errors", [])

        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "error": "Clinical assessment extraction confidence is below acceptable threshold or validation failed.",
                "overall_confidence": overall_score,
                "minimum_required_threshold": settings.MIN_CONFIDENCE_THRESHOLD,
                "flagged_fields": flagged,
                "notes": notes,
                "validation_errors": val_errors,
                "raw_transcription": transcription,
            },
        )

    # Return pure FirstAssessment JSON
    return result["assessment"]


@router.post(
    "",
    response_model=AssessmentSaveResponse,
    summary="EP2 — Save Assessment",
    status_code=status.HTTP_201_CREATED,
)
async def save_assessment(assessment: FirstAssessment):
    """
    Saves a parsed FirstAssessment object to MongoDB.
    Returns the saved assessment and its ID.
    """
    try:
        saved = await db.save_assessment(assessment)
        return AssessmentSaveResponse(
            id=saved["id"],
            assessment=saved["assessment"],
            created_at=saved["created_at"],
            message="Assessment saved successfully",
        )
    except Exception as e:
        logger.error(f"Error persisting assessment to MongoDB: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": f"Failed to save assessment: {str(e)}"},
        )


@router.get(
    "/{id}",
    summary="EP3 — Retrieve Assessment by ID",
    status_code=status.HTTP_200_OK,
)
async def get_assessment(id: str):
    """
    Retrieves a saved assessment from MongoDB using its ID.
    """
    doc = await db.get_assessment_by_id(id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": f"Assessment with ID '{id}' not found."},
        )
    return doc


@router.get(
    "",
    response_model=AssessmentListResponse,
    summary="EP4 — List Assessments",
    status_code=status.HTTP_200_OK,
)
async def list_assessments(
    date: Optional[str] = Query(None, description="Filter by exact date prefix (e.g. YYYY-MM-DD)"),
    start_date: Optional[str] = Query(None, description="Filter from start date ISO string"),
    end_date: Optional[str] = Query(None, description="Filter up to end date ISO string"),
    skip: int = Query(0, ge=0, description="Pagination skip offset"),
    limit: int = Query(50, ge=1, le=500, description="Pagination limit"),
):
    """
    Returns all saved assessments. Supports date filtering and pagination.
    """
    try:
        items = await db.list_assessments(
            date_str=date,
            start_date=start_date,
            end_date=end_date,
            skip=skip,
            limit=limit,
        )
        return AssessmentListResponse(
            total=len(items),
            items=items,
        )
    except Exception as e:
        logger.error(f"Error listing assessments: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": f"Failed to list assessments: {str(e)}"},
        )
