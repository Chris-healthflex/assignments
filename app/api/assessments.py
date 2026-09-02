"""FastAPI routes for clinical assessments."""

from datetime import datetime
from pathlib import Path
import tempfile
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status
from pydantic import BaseModel, Field

from app.api.deps import (
    get_assessment_repo,
    get_extraction_agent,
    get_transcriber,
)
from app.repositories.assessment_repo import (
    AssessmentNotFoundError,
    AssessmentRepository,
    InvalidAssessmentIdError,
    RepositoryException,
)
from app.schemas.assessment import FirstAssessment
from app.services.langgraph_agent import (
    ClinicalExtractionAgent,
    ExtractionState,
)
from app.services.transcriber import (
    AudioValidationError,
    TranscriptionError,
    WhisperTranscriber,
)

router = APIRouter(prefix="/assessments", tags=["Assessments"])


# ---------------------------------------------------------------------------
# API Response Models
# ---------------------------------------------------------------------------

class AssessmentSavedResponse(BaseModel):
    """Response model for saved assessment (EP2)."""

    id: str = Field(description="MongoDB ObjectId string identifier")
    message: str = Field(default="Assessment saved successfully")
    created_at: str = Field(description="ISO-8601 creation timestamp")
    assessment: FirstAssessment = Field(description="Persisted FirstAssessment data")


class AssessmentListResponse(BaseModel):
    """Response model for listing assessments (EP4)."""

    total: int = Field(description="Total number of matching assessments")
    skip: int = Field(default=0, description="Offset used for pagination")
    limit: int = Field(default=20, description="Limit used for pagination")
    items: List[FirstAssessment] = Field(description="List of FirstAssessment payloads")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post(
    "/parse",
    response_model=FirstAssessment,
    status_code=status.HTTP_200_OK,
    summary="EP1: Parse WAV audio into structured FirstAssessment",
    description="Upload a WAV audio recording, transcribe via Whisper, and extract clinical entities into FirstAssessment.",
    responses={
        200: {"description": "Successfully extracted FirstAssessment JSON."},
        400: {"description": "Invalid or empty audio upload."},
        422: {"description": "Extraction confidence below threshold or ungrounded fields detected."},
        500: {"description": "Transcription or processing error."},
    },
)
async def parse_assessment_audio(
    file: UploadFile = File(..., description="Multipart WAV audio recording"),
    transcriber: WhisperTranscriber = Depends(get_transcriber),
    agent: ClinicalExtractionAgent = Depends(get_extraction_agent),
) -> FirstAssessment:
    """EP1: Accept multipart WAV, transcribe with Whisper, run LangGraph, and return FirstAssessment."""
    # 1. Validate basic filename extension
    filename = file.filename or ""
    suffix = Path(filename).suffix.lower()
    if suffix and suffix not in {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".webm"}:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format '{suffix}'. Please upload a valid audio file (.wav).",
        )

    # 2. Read content safely and verify non-empty
    content = await file.read()
    if not content:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded audio file is empty (0 bytes).",
        )

    # 3. Stream to a temporary file for Whisper ingestion
    temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    temp_path = Path(temp_file.name)

    try:
        temp_path.write_bytes(content)
        temp_file.close()

        # 4. Transcribe audio with Whisper
        try:
            transcript = transcriber.transcribe(temp_path)
        except AudioValidationError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(exc),
            ) from exc
        except TranscriptionError as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Audio transcription failed: {str(exc)}",
            ) from exc

        # 5. Run LangGraph extraction workflow
        state: ExtractionState = agent.extract(transcript)

        # 6. Check confidence and grounding validation
        if not state.get("is_valid", False) or state.get("uncertain_fields") or state.get("validation_errors"):
            field_errors: List[Dict[str, Any]] = []

            for u in state.get("uncertain_fields", []):
                field_errors.append({
                    "field": u.get("field", "unknown"),
                    "message": u.get("reason", "Extracted value is not supported by transcript"),
                    "value": str(u.get("value", "")),
                })

            for err in state.get("validation_errors", []):
                field_errors.append({
                    "field": "transcript",
                    "message": str(err),
                })

            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=field_errors or [{"field": "general", "message": "Clinical extraction confidence below threshold."}],
            )

        # 7. Return clean production FirstAssessment
        final_assessment = state.get("final_assessment")
        if not final_assessment:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to construct FirstAssessment output.",
            )

        return final_assessment

    finally:
        # Guarantee removal of temporary file
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass


@router.post(
    "",
    response_model=AssessmentSavedResponse,
    status_code=status.HTTP_201_CREATED,
    summary="EP2: Save parsed FirstAssessment to MongoDB",
    description="Accepts a FirstAssessment JSON body and saves it to MongoDB.",
)
async def save_assessment(
    assessment: FirstAssessment,
    repo: AssessmentRepository = Depends(get_assessment_repo),
) -> AssessmentSavedResponse:
    """EP2: Save validated FirstAssessment payload to MongoDB."""
    try:
        doc = await repo.create(assessment)
        return AssessmentSavedResponse(
            id=doc.id,
            message="Assessment saved successfully",
            created_at=doc.created_at.isoformat(),
            assessment=doc.assessment,
        )
    except RepositoryException as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error while saving assessment: {str(exc)}",
        ) from exc


@router.get(
    "/{id}",
    response_model=FirstAssessment,
    status_code=status.HTTP_200_OK,
    summary="EP3: Retrieve saved assessment by ID",
    description="Retrieves a persisted FirstAssessment by its MongoDB ObjectId identifier.",
    responses={
        200: {"description": "FirstAssessment successfully retrieved."},
        404: {"description": "Assessment ID not found."},
    },
)
async def get_assessment_by_id(
    id: str,
    repo: AssessmentRepository = Depends(get_assessment_repo),
) -> FirstAssessment:
    """EP3: Retrieve saved FirstAssessment by MongoDB ID."""
    try:
        doc = await repo.get_by_id(id)
        if not doc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Assessment with ID '{id}' not found.",
            )
        # Returns the pure FirstAssessment production schema
        return doc.assessment
    except RepositoryException as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error while retrieving assessment: {str(exc)}",
        ) from exc


@router.get(
    "",
    response_model=AssessmentListResponse,
    status_code=status.HTTP_200_OK,
    summary="EP4: List all assessments with date filtering",
    description="List all assessments with optional start_date and end_date filtering and pagination.",
)
async def list_assessments(
    start_date: Optional[str] = Query(None, description="Start date filter (ISO-8601 format, e.g. 2026-09-01T00:00:00Z)"),
    end_date: Optional[str] = Query(None, description="End date filter (ISO-8601 format, e.g. 2026-09-02T23:59:59Z)"),
    skip: int = Query(0, ge=0, description="Number of items to skip"),
    limit: int = Query(20, ge=1, le=100, description="Maximum items to return"),
    repo: AssessmentRepository = Depends(get_assessment_repo),
) -> AssessmentListResponse:
    """EP4: List assessments, filter by date range, and return pure FirstAssessment items."""
    try:
        docs, total = await repo.list(
            start_date=start_date,
            end_date=end_date,
            skip=skip,
            limit=limit,
        )
        return AssessmentListResponse(
            total=total,
            skip=skip,
            limit=limit,
            items=[doc.assessment for doc in docs],
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid date parameter format: {str(exc)}",
        ) from exc
    except RepositoryException as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error while listing assessments: {str(exc)}",
        ) from exc
