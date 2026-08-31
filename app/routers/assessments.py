from __future__ import annotations

import os
import shutil
import uuid
from datetime import datetime, timezone
from typing import Optional

from bson import ObjectId
from bson.errors import InvalidId
from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.db import get_collection
from app.models.db_models import AssessmentDocument
from app.models.schema import FirstAssessment
from app.services.extraction_agent import run_extraction
from app.services.transcription import transcribe_wav

router = APIRouter(prefix="/assessments", tags=["assessments"])

ALLOWED_CONTENT_TYPES = {"audio/wav", "audio/x-wav", "audio/wave"}


@router.post("/parse")
async def parse_assessment(file: UploadFile = File(...)):
    """WAV upload -> transcription -> LangGraph extraction -> FirstAssessment JSON.

    Returns 422 with field-level detail if extraction confidence is below
    the configured threshold, per the brief. Does not persist — persistence
    is a separate explicit step via POST /assessments.
    """
    if file.content_type not in ALLOWED_CONTENT_TYPES and not file.filename.lower().endswith(".wav"):
        raise HTTPException(status_code=400, detail="Only WAV audio files are accepted.")

    settings = get_settings()
    os.makedirs(settings.upload_dir, exist_ok=True)
    tmp_path = os.path.join(settings.upload_dir, f"{uuid.uuid4().hex}.wav")

    try:
        with open(tmp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        transcription = transcribe_wav(tmp_path)
        if not transcription.text:
            raise HTTPException(status_code=422, detail="Transcription produced no speech content.")

        result = run_extraction(transcription.text)

    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Pipeline error: {e}") from e
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    if result.overall_confidence < settings.min_overall_confidence or result.low_confidence_fields:
        return JSONResponse(
            status_code=422,
            content={
                "detail": "Extraction confidence below threshold; review flagged fields before saving.",
                "overall_confidence": result.overall_confidence,
                "low_confidence_fields": result.low_confidence_fields,
                "field_confidences": [fc.model_dump() for fc in result.field_confidences],
                "assessment": result.assessment.model_dump(),
                "transcript": result.transcript,
                "audio_filename": file.filename,
            },
        )

    return {
        "assessment": result.assessment.model_dump(),
        "transcript": result.transcript,
        "overall_confidence": result.overall_confidence,
        "low_confidence_fields": result.low_confidence_fields,
        "audio_filename": file.filename,
    }


@router.post("")
async def save_assessment(payload: dict):
    """Persist a previously parsed result to MongoDB.

    Expects the shape returned by /assessments/parse: at minimum
    {"assessment": {...FirstAssessment...}, "transcript": "...", "audio_filename": "..."}
    """
    try:
        assessment = FirstAssessment.model_validate(payload["assessment"])
    except KeyError:
        raise HTTPException(status_code=422, detail="Missing 'assessment' field.")
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=422, detail=f"Invalid assessment payload: {e}") from e

    doc = AssessmentDocument(
        assessment=assessment,
        transcript=payload.get("transcript", ""),
        audio_filename=payload.get("audio_filename"),
        overall_confidence=payload.get("overall_confidence", 1.0),
        low_confidence_fields=payload.get("low_confidence_fields", []),
    )

    collection = get_collection()
    result = await collection.insert_one(doc.to_mongo())
    return {"id": str(result.inserted_id)}


@router.get("/{assessment_id}")
async def get_assessment(assessment_id: str):
    try:
        oid = ObjectId(assessment_id)
    except InvalidId:
        raise HTTPException(status_code=400, detail="Invalid assessment id.")

    collection = get_collection()
    doc = await collection.find_one({"_id": oid})
    if not doc:
        raise HTTPException(status_code=404, detail="Assessment not found.")

    doc["_id"] = str(doc["_id"])
    return doc


@router.get("")
async def list_assessments(
    date_from: Optional[str] = Query(default=None, description="ISO date, e.g. 2026-01-01"),
    date_to: Optional[str] = Query(default=None, description="ISO date, e.g. 2026-12-31"),
    limit: int = Query(default=50, le=200),
):
    query: dict = {}
    if date_from or date_to:
        date_filter = {}
        if date_from:
            date_filter["$gte"] = datetime.fromisoformat(date_from).replace(tzinfo=timezone.utc)
        if date_to:
            date_filter["$lte"] = datetime.fromisoformat(date_to).replace(tzinfo=timezone.utc)
        query["created_at"] = date_filter

    collection = get_collection()
    cursor = collection.find(query).sort("created_at", -1).limit(limit)
    results = []
    async for doc in cursor:
        doc["_id"] = str(doc["_id"])
        results.append(doc)
    return {"count": len(results), "results": results}
