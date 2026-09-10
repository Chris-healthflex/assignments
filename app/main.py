from __future__ import annotations
import time
import logging
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, UploadFile, File, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import settings
from app.db.mongo import init_db, close_db
from app.models.assessment import FirstAssessment
from app.models.internal import AssessmentRecord, ConfidenceReport, AssessmentListResponse
from app.repositories.assessment_repository import AssessmentRepository
from app.services.audio import AudioValidationError, validate_wav_header
from app.services.transcription import transcribe
from app.services.extraction import build_pipeline_graph

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    try:
        await init_db()
        logger.info("Connected to MongoDB successfully.")
    except Exception as exc:
        logger.warning("MongoDB connection failed at startup (%s). Ensure MongoDB is running.", exc)
    yield
    # Shutdown
    await close_db()


app = FastAPI(
    title="Stance Health Clinical Assessment Pipeline",
    version="1.0.0",
    description="WAV-to-structured-JSON clinical assessment form filler with deterministic grounding and confidence fusion.",
    lifespan=lifespan,
)

# CORS Middleware to support frontend consumption
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=[
        "X-Extraction-Confidence",
        "X-Extraction-Model",
        "X-Pipeline-Latency-Ms",
    ],
)

_pipeline_graph = None


def get_pipeline():
    global _pipeline_graph
    if _pipeline_graph is None:
        _pipeline_graph = build_pipeline_graph()
    return _pipeline_graph


@app.get("/health")
async def health():
    return {"status": "healthy", "time": datetime.now(timezone.utc).isoformat()}


@app.post(
    "/assessments/parse",
    response_model=FirstAssessment,
    responses={
        200: {"description": "Strict FirstAssessment JSON response body without extra keys"},
        400: {"description": "Invalid or non-WAV audio file"},
        422: {"description": "Confidence below threshold, returning ConfidenceReport with flags", "model": ConfidenceReport},
        500: {"description": "Transcription or LLM extraction failure"},
    },
)
async def parse_assessment_audio(file: UploadFile = File(...)):
    """
    Parses an uploaded WAV audio file into a strict FirstAssessment JSON object.
    Confidence data and latency metrics are strictly delivered via HTTP headers.
    """
    t0 = time.time()

    # 1. Verify audio format
    content = await file.read()
    try:
        validate_wav_header(content)
    except AudioValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Audio validation error: {str(exc)}",
        )

    # 2. Whisper transcription
    t_whisper_start = time.time()
    try:
        transcript_result = transcribe(content, model_size=settings.WHISPER_MODEL)
    except Exception as exc:
        logger.error("Whisper transcription failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Transcription failed: {str(exc)}",
        )
    t_whisper_ms = int((time.time() - t_whisper_start) * 1000)

    # 3. LangGraph extraction pipeline (extract -> ground -> normalize -> audit)
    t_pipeline_start = time.time()
    try:
        pipeline = get_pipeline()
        result_state = pipeline.invoke({
            "transcript": transcript_result.text,
            "segments": transcript_result.segments,
        })
    except Exception as exc:
        logger.error("Pipeline extraction failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Extraction failed: {str(exc)}",
        )
    t_pipeline_ms = int((time.time() - t_pipeline_start) * 1000)
    t_total_ms = int((time.time() - t0) * 1000)

    assessment: FirstAssessment = result_state["assessment"]
    confidence: ConfidenceReport = result_state["confidence_report"]

    model_name = getattr(settings, f"{settings.LLM_PROVIDER.upper()}_MODEL", settings.LLM_PROVIDER)
    headers = {
        "X-Extraction-Confidence": f"{confidence.overall:.2f}",
        "X-Extraction-Model": str(model_name),
        "X-Pipeline-Latency-Ms": f"whisper={t_whisper_ms},pipeline={t_pipeline_ms},total={t_total_ms}",
    }

    # If confidence is below threshold, return 422 with ConfidenceReport
    if not confidence.passed:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=confidence.model_dump(),
            headers=headers,
        )

    # Success: return exact FirstAssessment body
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content=assessment.model_dump(),
        headers=headers,
    )


@app.post(
    "/assessments",
    response_model=AssessmentRecord,
    status_code=status.HTTP_201_CREATED,
)
async def save_assessment(assessment: FirstAssessment):
    """Saves a validated FirstAssessment into MongoDB and returns an AssessmentRecord."""
    try:
        repo = AssessmentRepository()
        record = await repo.create(assessment)
        return record
    except Exception as exc:
        logger.error("Failed to save assessment to MongoDB: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to persist assessment: {str(exc)}",
        )


@app.get(
    "/assessments/{assessment_id}",
    response_model=AssessmentRecord,
)
async def get_assessment(assessment_id: str):
    """Retrieves an AssessmentRecord by ID."""
    try:
        repo = AssessmentRepository()
        record = await repo.get_by_id(assessment_id)
        if not record:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Assessment '{assessment_id}' not found",
            )
        return record
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to fetch assessment: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database query error: {str(exc)}",
        )


@app.get(
    "/assessments",
    response_model=AssessmentListResponse,
)
async def list_assessments(
    from_date: Optional[str] = Query(None, alias="from", description="ISO datetime string start"),
    to_date: Optional[str] = Query(None, alias="to", description="ISO datetime string end"),
    limit: int = Query(20, ge=1, le=100),
    skip: int = Query(0, ge=0),
):
    """Lists saved assessments with pagination and optional date-range filtering."""
    parsed_from = None
    parsed_to = None

    if from_date:
        try:
            parsed_from = datetime.fromisoformat(from_date.replace("Z", "+00:00"))
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid 'from' date format. Use ISO format.")

    if to_date:
        try:
            parsed_to = datetime.fromisoformat(to_date.replace("Z", "+00:00"))
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid 'to' date format. Use ISO format.")

    if parsed_from and parsed_to and parsed_from > parsed_to:
        raise HTTPException(status_code=400, detail="'from' date cannot be after 'to' date.")

    try:
        repo = AssessmentRepository()
        items, total = await repo.list_all(
            from_date=parsed_from,
            to_date=parsed_to,
            skip=skip,
            limit=limit,
        )
        return AssessmentListResponse(
            items=items,
            total=total,
            limit=limit,
            skip=skip,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to list assessments: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database query error: {str(exc)}",
        )
