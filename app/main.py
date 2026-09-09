from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.agent import run_extraction_pipeline
from app.config import settings
from app.db import get_assessment_by_id, list_assessments, save_assessment
from app.schemas import (
    AssessmentCreateRequest,
    AssessmentRecord,
    FirstAssessment,
)
from app.transcription import transcribe_audio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("stance_assessment")

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Transforms clinician-patient audio into structured FirstAssessment JSON.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Extraction-Confidence", "X-Extraction-Flags"],
)


@app.get("/health", tags=["Health"])
async def health_check():
    """Service health and readiness check."""
    return {
        "status": "healthy",
        "version": settings.APP_VERSION,
        "app": settings.APP_NAME,
    }


@app.post(
    "/assessments/parse",
    response_model=FirstAssessment,
    tags=["Assessments"],
    summary="Transcribe WAV and extract structured FirstAssessment JSON",
    responses={
        200: {"description": "Structured FirstAssessment matching exact frontend schema"},
        400: {"description": "Invalid, corrupted or non-WAV audio"},
        422: {"description": "Extraction confidence below threshold or core fields missing"},
        500: {"description": "Internal server / pipeline error"},
    },
)
async def parse_assessment_audio(
    file: UploadFile = File(..., description="WAV audio recording of clinical session"),
    session_date: Optional[str] = Form(None, description="ISO session date to resolve relative timelines"),
    save: bool = Form(False, description="Persist extracted assessment to MongoDB"),
):
    """
    Core Pipeline Endpoint:
    1. Validates WAV format and reads audio bytes.
    2. Transcribes dialogue via Whisper (in-process without requiring ffmpeg).
    3. Runs LangGraph extraction agent + deterministic anti-hallucination audit.
    4. Enforces strict FirstAssessment schema.
    5. Returns exact JSON body with confidence metrics in headers (or 422 if low confidence).
    """
    # 1. Validate file extension / mime
    if not file.filename.lower().endswith(".wav"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file format. Uploaded file must be a .wav recording.",
        )

    try:
        audio_bytes = await file.read()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Could not read uploaded audio: {str(e)}",
        )

    if not audio_bytes or len(audio_bytes) < 44:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded WAV file is empty or corrupted.",
        )

    # 2. Transcribe Audio
    try:
        transcript = transcribe_audio(audio_bytes)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Transcription failure: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Transcription failed: {str(e)}",
        )

    # 3. Extract & Audit
    try:
        result = run_extraction_pipeline(transcript, session_date=session_date)
    except Exception as e:
        logger.error(f"Extraction failure: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Extraction agent failed: {str(e)}",
        )

    # 4. Handle Low Confidence (< CONFIDENCE_THRESHOLD or missing core fields)
    if result.low_confidence:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "detail": {
                    "message": "Extraction confidence below threshold or core fields missing",
                    "overall_confidence": result.overall_confidence,
                    "fields": [f.model_dump() for f in result.flags],
                }
            },
        )

    # 5. Optional Save
    if save:
        await save_assessment(
            assessment=result.assessment,
            meta={
                "source_file": file.filename,
                "transcript": result.transcript,
                "overall_confidence": result.overall_confidence,
                "flags": [f.model_dump() for f in result.flags],
            },
        )

    headers = {
        "X-Extraction-Confidence": str(result.overall_confidence),
        "X-Extraction-Flags": json.dumps([f.model_dump() for f in result.flags]),
    }

    return JSONResponse(
        content=result.assessment.model_dump(),
        headers=headers,
    )


@app.post(
    "/assessments",
    response_model=AssessmentRecord,
    status_code=status.HTTP_201_CREATED,
    tags=["Assessments"],
    summary="Save a structured FirstAssessment to MongoDB",
)
async def create_assessment(payload: AssessmentCreateRequest):
    """Saves a structured FirstAssessment document to MongoDB."""
    saved = await save_assessment(payload.assessment, meta=payload.meta)
    return saved


@app.get(
    "/assessments/{assessment_id}",
    response_model=AssessmentRecord,
    tags=["Assessments"],
    summary="Retrieve an assessment by its MongoDB ID",
)
async def get_assessment(assessment_id: str):
    """Retrieves an existing assessment by its MongoDB ObjectID."""
    record = await get_assessment_by_id(assessment_id)
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Assessment with ID '{assessment_id}' not found.",
        )
    return record


@app.get(
    "/assessments",
    response_model=list[AssessmentRecord],
    tags=["Assessments"],
    summary="List assessments with optional date filters and pagination",
)
async def list_all_assessments(
    from_date: Optional[datetime] = Query(None, alias="from", description="Filter from createdAt (ISO)"),
    to_date: Optional[datetime] = Query(None, alias="to", description="Filter to createdAt (ISO)"),
    limit: int = Query(50, ge=1, le=100),
    skip: int = Query(0, ge=0),
):
    """Lists saved assessments ordered newest first."""
    if from_date and to_date and from_date > to_date:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="'from' date cannot be after 'to' date.",
        )
    return await list_assessments(from_date=from_date, to_date=to_date, limit=limit, skip=skip)
