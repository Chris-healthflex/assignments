"""
FastAPI service.

POST /assessments/parse   multipart WAV -> FirstAssessment JSON (exact schema)
POST /assessments         save a FirstAssessment to MongoDB
GET  /assessments/{id}    fetch one
GET  /assessments         list, filterable by date range

Confidence contract for /assessments/parse:
  * every flagged field is echoed in the `X-Extraction-Flags` response header
    (JSON) so the frontend can highlight fields for clinician review without
    polluting the body, which must be an exact schema match;
  * if a CORE field or the overall confidence is below CONFIDENCE_THRESHOLD
    the endpoint returns HTTP 422 with field-level detail instead of a body
    the frontend might silently trust.
"""
from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile, status
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse

from . import db
from .agent import run_extraction
from .config import settings
from .schemas import FirstAssessment
from .transcription import AudioDecodeError, transcribe

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("stance")


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        await db.ensure_indexes()
    except Exception as e:  # keep the API up even if Mongo is down; writes will fail loudly
        log.warning("MongoDB not reachable at startup: %s", e)
    yield
    await db.close()


app = FastAPI(
    title="Stance Health — Voice → FirstAssessment",
    version="1.0.0",
    description="Transcribes a clinician–patient WAV session and returns a FirstAssessment JSON.",
    lifespan=lifespan,
)


@app.exception_handler(Exception)
async def _unhandled(request: Request, exc: Exception):
    log.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"detail": f"{type(exc).__name__}: {exc}"})


@app.get("/health", tags=["ops"])
async def health():
    return {"status": "ok", "whisper": settings.whisper_backend, "llm": f"{settings.llm_provider}:{settings.llm_model}"}


# --------------------------------------------------------------------------- EP1
@app.post(
    "/assessments/parse",
    response_model=FirstAssessment,
    tags=["assessments"],
    summary="Multipart WAV → FirstAssessment JSON",
    responses={422: {"description": "Extraction confidence below threshold (field-level detail in body)"}},
)
async def parse_assessment(
    file: UploadFile = File(..., description="WAV recording of the session"),
    session_date: Optional[str] = Form(None, description="Optional ISO date (YYYY-MM-DD) used to resolve relative dates"),
    save: bool = Form(False, description="If true, also persist the result to MongoDB"),
):
    if not (file.filename or "").lower().endswith(".wav") and file.content_type not in ("audio/wav", "audio/x-wav", "audio/wave"):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="Only WAV files are accepted")

    wav_bytes = await file.read()
    if not wav_bytes:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="Empty upload")

    try:
        transcript = await run_in_threadpool(transcribe, wav_bytes)
    except AudioDecodeError:
        # The upload is named .wav but is not decodable audio -> client error,
        # not a server fault. Deliberately does not echo the decoder's message,
        # which contains buffer reprs rather than anything actionable.
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="File is not a readable WAV audio stream")
    except Exception as e:
        log.exception("Transcription failed")
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Transcription failed: {type(e).__name__}")
    if not transcript.strip():
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, detail=[{"loc": ["transcript"], "msg": "No speech detected in audio", "type": "empty_transcript"}])

    try:
        result = await run_in_threadpool(run_extraction, transcript, session_date)
    except Exception as e:
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Extraction failed: {e}")

    if result.low_confidence:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "message": f"Extraction confidence below threshold ({settings.confidence_threshold})",
                "overall_confidence": result.overall_confidence,
                "fields": [f.model_dump() for f in result.flags],
            },
        )

    if save:
        await db.save_assessment(
            result.assessment,
            db.AssessmentMeta(sourceFile=file.filename or "", transcript=transcript,
                              flags=result.flags, overallConfidence=result.overall_confidence),
        )

    headers = {
        "X-Extraction-Confidence": str(result.overall_confidence),
        "X-Extraction-Flags": json.dumps([f.model_dump() for f in result.flags]),
    }
    return JSONResponse(content=result.assessment.model_dump(), headers=headers)


# --------------------------------------------------------------------------- EP2
@app.post("/assessments", response_model=db.StoredAssessment, status_code=201, tags=["assessments"],
          summary="Save a parsed FirstAssessment to MongoDB")
async def create_assessment(body: db.SaveAssessmentRequest):
    return await db.save_assessment(body.assessment, body.meta)


# --------------------------------------------------------------------------- EP3
@app.get("/assessments/{assessment_id}", response_model=db.StoredAssessment, tags=["assessments"],
         summary="Retrieve a saved assessment by ID")
async def read_assessment(assessment_id: str):
    doc = await db.get_assessment(assessment_id)
    if doc is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=f"Assessment {assessment_id} not found")
    return doc


# --------------------------------------------------------------------------- EP4
@app.get("/assessments", response_model=list[db.StoredAssessment], tags=["assessments"],
         summary="List assessments, filterable by createdAt date range")
async def list_assessments(
    from_date: Optional[datetime] = Query(None, alias="from", description="ISO datetime, inclusive"),
    to_date: Optional[datetime] = Query(None, alias="to", description="ISO datetime, inclusive"),
    limit: int = Query(100, ge=1, le=500),
    skip: int = Query(0, ge=0),
):
    if from_date and to_date and from_date > to_date:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="`from` must be <= `to`")
    return await db.list_assessments(from_date, to_date, limit, skip)
