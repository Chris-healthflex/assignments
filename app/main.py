"""FastAPI application: audio -> transcript -> structured FirstAssessment."""

from __future__ import annotations

import logging
import shutil
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile

from app import db
from app.config import get_settings
from app.extraction import ExtractionFailed, extract
from app.schemas import (
    AssessmentResponse,
    ExtractionMeta,
    ExtractRequest,
    FirstAssessment,
    TranscriptionResult,
)
from app.transcription import TranscriptionError, transcribe

settings = get_settings()
logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await db.ensure_indexes()
    yield
    await db.close()


app = FastAPI(title=settings.app_name, version="0.1.0", lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, object]:
    return {"status": "ok", "mongo": await db.ping()}


@app.post("/transcribe", response_model=TranscriptionResult)
async def transcribe_endpoint(file: UploadFile = File(...)) -> TranscriptionResult:
    """Transcribe an uploaded consultation recording."""
    suffix = Path(file.filename or "audio").suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)
    try:
        return transcribe(tmp_path)
    except TranscriptionError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    finally:
        tmp_path.unlink(missing_ok=True)


@app.post("/extract", response_model=AssessmentResponse)
async def extract_endpoint(payload: ExtractRequest) -> AssessmentResponse:
    """Turn a transcript into a persisted FirstAssessment."""
    try:
        extracted, warnings = extract(payload.transcript)
    except ExtractionFailed as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    assessment = FirstAssessment(
        session_id=payload.session_id,
        patient=extracted.patient,
        complaints=extracted.complaints,
        medical_history=extracted.medical_history,
        lifestyle=extracted.lifestyle,
        goals=extracted.goals,
        clinician_notes=extracted.clinician_notes,
        transcript=payload.transcript,
        meta=ExtractionMeta(
            model=settings.extraction_model,
            unresolved_fields=extracted.unresolved_fields,
            warnings=warnings,
        ),
    )
    assessment_id = await db.save_assessment(assessment)
    assessment.assessment_id = assessment_id
    return AssessmentResponse(assessment_id=assessment_id, assessment=assessment)


@app.get("/assessments/{assessment_id}", response_model=FirstAssessment)
async def get_assessment(assessment_id: str) -> FirstAssessment:
    assessment = await db.get_assessment(assessment_id)
    if assessment is None:
        raise HTTPException(status_code=404, detail="Assessment not found")
    return assessment


@app.get("/assessments", response_model=list[FirstAssessment])
async def list_assessments(limit: int = 20) -> list[FirstAssessment]:
    return await db.list_assessments(limit=limit)
