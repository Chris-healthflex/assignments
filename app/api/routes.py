"""The four assessment endpoints."""
from __future__ import annotations

import asyncio
import logging
import shutil
import tempfile
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status
from pydantic import BaseModel

from app.agent.graph import run_extraction, _default_llm
from app.config import get_settings
from app.db.repository import AssessmentRepository, InvalidAssessmentId
from app.models.assessment import FirstAssessment
from app.models.internal import PipelineResult, StoredAssessment, LowConfidenceErrorResponse
from app.transcription.whisper_service import (
    TranscriptionError,
    WhisperTranscriber,
)

logger = logging.getLogger(__name__)
router = APIRouter(tags=["assessments"])

_transcriber = WhisperTranscriber()


def get_transcriber():
    """Overridable dependency so tests can inject a fake transcriber."""
    return _transcriber

def get_llm():
    """Overridable dependency so tests can inject a fake LLM."""
    return _default_llm()


def get_repository() -> AssessmentRepository:
    """Overridable dependency so tests can inject a fake collection."""
    return AssessmentRepository()


class SaveAssessmentRequest(BaseModel):
    assessment: FirstAssessment
    source_transcript: Optional[str] = None


def _raise_low_confidence(result: PipelineResult) -> None:
    """Turn failed confidence checks into a 422 with field-level detail."""
    settings = get_settings()
    raise HTTPException(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        detail={
            "message": "Extraction confidence below threshold",
            "threshold": settings.confidence_threshold,
            "fields": [f.model_dump() for f in result.low_confidence_fields],
        },
    )


@router.post("/assessments/parse", response_model=FirstAssessment, responses={422: {"model": LowConfidenceErrorResponse}})
async def parse_assessment(
    file: UploadFile = File(...),
    transcriber=Depends(get_transcriber),
    llm=Depends(get_llm),
) -> FirstAssessment:
    """WAV upload -> transcription -> LangGraph extraction -> FirstAssessment JSON."""
    settings = get_settings()

    if not file.filename or not file.filename.lower().endswith(".wav"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only .wav files are accepted.",
        )

    tmp_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = Path(tmp.name)

        if tmp_path.stat().st_size > settings.max_upload_bytes:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File exceeds {settings.max_upload_bytes} bytes.",
            )

        try:
            transcript = await asyncio.to_thread(transcriber.transcribe, tmp_path)
        except TranscriptionError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
            ) from exc

        result = await asyncio.to_thread(run_extraction, transcript, llm=llm)

        if result.error:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY, detail=result.error
            )
        if result.low_confidence_fields:
            _raise_low_confidence(result)

        return result.assessment
    finally:
        await file.close()
        if tmp_path and tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


@router.post(
    "/assessments",
    response_model=StoredAssessment,
    status_code=status.HTTP_201_CREATED,
)
async def create_assessment(
    payload: SaveAssessmentRequest,
    repo: AssessmentRepository = Depends(get_repository),
) -> StoredAssessment:
    """Persist a parsed assessment."""
    return await repo.save(payload.assessment, payload.source_transcript)


@router.get("/assessments", response_model=List[StoredAssessment])
async def list_assessments(
    date: Optional[str] = Query(None, description="Exact day, YYYY-MM-DD"),
    from_date: Optional[str] = Query(None, description="Range start, YYYY-MM-DD"),
    to_date: Optional[str] = Query(None, description="Range end, YYYY-MM-DD"),
    limit: int = Query(100, ge=1, le=500),
    last_id: Optional[str] = Query(None, description="Cursor for pagination"),
    repo: AssessmentRepository = Depends(get_repository),
) -> List[StoredAssessment]:
    """List assessments, newest first, optionally filtered by date."""
    try:
        return await repo.list(
            date=date, from_date=from_date, to_date=to_date, limit=limit, last_id=last_id
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid date filter, expected YYYY-MM-DD: {exc}",
        ) from exc


@router.get("/assessments/{assessment_id}", response_model=StoredAssessment)
async def get_assessment(
    assessment_id: str,
    repo: AssessmentRepository = Depends(get_repository),
) -> StoredAssessment:
    """Retrieve one assessment by id."""
    try:
        stored = await repo.get_by_id(assessment_id)
    except InvalidAssessmentId as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"'{assessment_id}' is not a valid assessment id.",
        ) from exc
    if stored is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Assessment '{assessment_id}' not found.",
        )
    return stored
