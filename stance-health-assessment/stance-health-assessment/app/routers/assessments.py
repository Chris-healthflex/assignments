from __future__ import annotations

import os
import tempfile
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from pydantic import ValidationError

from app.agent.graph import run_pipeline
from app.db.models import AssessmentRecord, save_assessment, get_assessment, list_assessments
from app.schemas.first_assessment import FirstAssessment, ExtractionEnvelope

router = APIRouter(prefix="/assessments", tags=["assessments"])

CONFIDENCE_THRESHOLD = float(os.getenv("EXTRACTION_CONFIDENCE_THRESHOLD", "0.4"))


@router.post("/parse")
async def parse_assessment(file: UploadFile = File(...)) -> dict:
    """WAV -> transcribe -> extract -> FirstAssessment JSON (+ metadata)."""
    if not file.filename.lower().endswith(".wav"):
        raise HTTPException(status_code=400, detail="Only .wav files are supported")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        envelope_dict = run_pipeline(tmp_path)
    except FileNotFoundError:
        raise HTTPException(status_code=400, detail="Could not read uploaded audio file")
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=f"Transcription/extraction backend error: {e}")
    finally:
        os.remove(tmp_path)

    if envelope_dict.get("overall_confidence", 0.0) < CONFIDENCE_THRESHOLD:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Extraction confidence below threshold; result not auto-accepted.",
                "overall_confidence": envelope_dict.get("overall_confidence"),
                "extraction_flags": envelope_dict.get("extraction_flags", []),
                "assessment": envelope_dict.get("assessment"),
            },
        )

    envelope_dict["source_audio_filename"] = file.filename
    return envelope_dict


@router.post("")
async def create_assessment(payload: dict) -> dict:
    """Persist a parsed result (typically the output of /parse) to MongoDB."""
    try:
        assessment = FirstAssessment(**payload["assessment"])
    except (KeyError, ValidationError) as e:
        raise HTTPException(status_code=422, detail=f"Invalid assessment payload: {e}")

    record = AssessmentRecord(
        assessment=assessment,
        overall_confidence=payload.get("overall_confidence", 1.0),
        extraction_flags=payload.get("extraction_flags", []),
        source_audio_filename=payload.get("source_audio_filename"),
    )
    saved_id = await save_assessment(record)
    return {"id": saved_id}


@router.get("/{assessment_id}")
async def read_assessment(assessment_id: str) -> dict:
    doc = await get_assessment(assessment_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="Assessment not found")
    return doc


@router.get("")
async def list_all_assessments(
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None),
    limit: int = Query(50, le=200),
    skip: int = Query(0, ge=0),
) -> list[dict]:
    return await list_assessments(start_date=start_date, end_date=end_date, limit=limit, skip=skip)
