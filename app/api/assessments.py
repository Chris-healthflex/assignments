from __future__ import annotations

from datetime import datetime
from typing import Any

from fastapi import APIRouter, HTTPException, Query, UploadFile, File, status

from app.schemas.assessment import FirstAssessment
from app.services.assessment_service import (
    ConfidenceTooLowError,
    get_assessment,
    list_assessments,
    parse_wav_to_assessment,
    save_assessment,
)
from app.services.transcription import TranscriptionError

router = APIRouter(prefix="/assessments", tags=["assessments"])


@router.post("/parse", response_model=FirstAssessment)
async def parse_assessment(file: UploadFile = File(...)) -> FirstAssessment:
    """
    EP1 — Accept a WAV recording, transcribe it with Whisper, run it through
    the LangGraph clinical extraction agent, and return the structured
    FirstAssessment JSON.

    Returns HTTP 422 with field-level detail if extraction confidence for
    any field is below the configured threshold (never hallucinates
    clinical values, scores, or dates to fill the gap).
    """
    if file.filename and not file.filename.lower().endswith(".wav"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only .wav files are supported.",
        )

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    try:
        return parse_wav_to_assessment(file_bytes, file.filename or "audio.wav")
    except TranscriptionError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except ConfidenceTooLowError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "message": "Extraction confidence below threshold for one or more fields.",
                "fields": exc.low_confidence_fields,
            },
        ) from exc


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_assessment(assessment: FirstAssessment) -> dict[str, str]:
    """EP2 — Persist an already-parsed FirstAssessment to MongoDB."""
    assessment_id = await save_assessment(assessment)
    return {"id": assessment_id}


@router.get("/{assessment_id}")
async def read_assessment(assessment_id: str) -> dict[str, Any]:
    """EP3 — Retrieve a saved assessment by ID."""
    document = await get_assessment(assessment_id)
    if document is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Assessment not found.")
    return document


@router.get("")
async def list_assessments_endpoint(
    date_from: datetime | None = Query(default=None, description="Filter: created on/after this ISO date"),
    date_to: datetime | None = Query(default=None, description="Filter: created on/before this ISO date"),
) -> list[dict[str, Any]]:
    """EP4 — List all saved assessments, optionally filtered by date range."""
    return await list_assessments(date_from=date_from, date_to=date_to)
