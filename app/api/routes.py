"""The four REST endpoints (D1).

EP1  POST /assessments/parse   multipart WAV -> FirstAssessment JSON
EP2  POST /assessments         persist a parsed result
EP3  GET  /assessments/{id}    retrieve by id
EP4  GET  /assessments         list, filterable by date

Whisper and the LLM are synchronous and CPU/GPU bound, so they run in a
threadpool. Calling them directly would block the event loop for the full
two-minute pipeline and stall every other request, including /health.

Descriptions here are written to render as documentation: a reviewer may
arrive at /docs rather than at the interface on /, so these strings are a
deliverable rather than an afterthought.
"""

from __future__ import annotations

import logging
import os
import tempfile
from datetime import datetime

from fastapi import APIRouter, File, HTTPException, Query, Request, UploadFile, status
from fastapi.concurrency import run_in_threadpool

from app.api.schemas import (
    AssessmentListResponse,
    ConfidenceInfo,
    ErrorResponse,
    HealthResponse,
    LowConfidenceDetail,
    ParseResponse,
    SaveAssessmentRequest,
    SavedAssessmentResponse,
    TranscriptInfo,
)
from app.config import get_settings
from app.db import client as db_client
from app.db import repository as repo
from app.db.models import AssessmentMetadata
from app.extraction.graph import extract_assessment
from app.extraction.llm import LLMUnavailableError
from app.transcription.audio_io import InvalidAudioError
from app.transcription.whisper_service import (
    EmptyTranscriptError,
    TranscriptionError,
    get_transcriber,
)

logger = logging.getLogger(__name__)

router = APIRouter()

#: Read the upload in chunks so a large file never lands in memory whole.
_CHUNK = 1024 * 1024


async def _save_upload(upload: UploadFile, max_bytes: int) -> str:
    """Stream an upload to a temp file, enforcing the size cap as we go."""
    if not upload.filename:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "No file was uploaded.")

    handle = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    written = 0
    try:
        while chunk := await upload.read(_CHUNK):
            written += len(chunk)
            if written > max_bytes:
                raise HTTPException(
                    status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    f"Upload exceeds the {max_bytes // (1024 * 1024)} MB limit.",
                )
            handle.write(chunk)
    except HTTPException:
        handle.close()
        os.unlink(handle.name)
        raise
    finally:
        if not handle.closed:
            handle.close()

    if written == 0:
        os.unlink(handle.name)
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "The uploaded file is empty.")

    return handle.name


@router.post(
    "/assessments/parse",
    response_model=None,
    summary="EP1 - Parse a WAV recording into a FirstAssessment",
    description=(
        "Upload a WAV recording of a clinician-patient session. The pipeline "
        "transcribes it with Whisper, extracts clinical entities with a "
        "LangGraph agent, verifies every value against the transcript, and "
        "returns the result in the exact FirstAssessment schema.\n\n"
        "**Anything that cannot be traced to the transcript is returned as an "
        "empty string and reported in `flaggedFields`.** Values are never "
        "guessed - a blank field is the correct answer for anything the "
        "recording did not cover.\n\n"
        "`flaggedFields` distinguishes three cases: `not_stated` (the recording "
        "never covered it), `rejected` (the model produced a value that failed "
        "verification and was discarded), and `possibly_missed` (a measurement "
        "the recording states that reached no field - the model dropped it).\n\n"
        "Pass `envelope=false` to receive the bare FirstAssessment object with "
        "no surrounding metadata.\n\n"
        "This is a slow endpoint: roughly 25 s of transcription plus around "
        "two minutes of extraction for a two-minute recording on local models."
    ),
    responses={
        400: {
            "model": ErrorResponse,
            "description": "Not a readable PCM WAV file, or it contains no speech",
        },
        413: {"model": ErrorResponse, "description": "Upload exceeds the size limit"},
        422: {
            "model": LowConfidenceDetail,
            "description": "Extraction confidence below threshold, with field-level detail",
        },
        503: {"model": ErrorResponse, "description": "Whisper or the LLM provider is unavailable"},
    },
    tags=["assessments"],
)
async def parse_assessment(
    request: Request,
    file: UploadFile = File(..., description="WAV recording of the session"),
    envelope: bool = Query(
        True,
        description="Return the assessment wrapped with transcript and flags. "
        "Set false for the bare FirstAssessment object.",
    ),
    save: bool = Query(False, description="Also persist the result to MongoDB."),
):
    settings = get_settings()
    path = await _save_upload(file, settings.max_upload_bytes)

    try:
        # S2 - transcribe
        try:
            transcript = await run_in_threadpool(get_transcriber().transcribe, path)
        except InvalidAudioError as exc:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc
        except EmptyTranscriptError as exc:
            # The service worked; the upload has no speech in it. Ordering
            # matters - this subclasses TranscriptionError and must be caught
            # first, or a silent recording reports the service as down.
            raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc
        except TranscriptionError as exc:
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc

        # S3, S4, S5 - extract, map, flag
        try:
            result = await run_in_threadpool(extract_assessment, transcript.text)
        except LLMUnavailableError as exc:
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc

        report = result["confidence"]
        assessment = result["assessment"]

        timings = dict(result["timings"])
        timings["transcribe"] = transcript.transcribeSeconds

        # The brief: 422 with field-level detail below the confidence threshold.
        if not report.meetsThreshold:
            logger.info(
                "Rejecting parse: confidence %.2f below threshold %.2f",
                report.overall, report.threshold,
            )
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_ENTITY,
                LowConfidenceDetail(
                    message=(
                        f"Extraction confidence {report.overall:.2f} is below the "
                        f"required threshold {report.threshold:.2f}. The recording may "
                        "be unclear or may not contain a full assessment."
                    ),
                    confidence=report.overall,
                    threshold=report.threshold,
                    fields=report.flaggedFields,
                    transcript=transcript.text,
                    assessment=assessment.model_dump(),
                ).model_dump(),
            )

        if save:
            await repo.save(
                assessment,
                AssessmentMetadata(
                    sourceFilename=file.filename or "",
                    transcript=transcript.text,
                    transcriptLanguage=transcript.language,
                    audioDurationSeconds=transcript.durationSeconds,
                    whisperModel=transcript.model,
                    whisperBackend=transcript.backend,
                    llmProvider=settings.llm_provider,
                    llmModel=settings.default_llm_model,
                    confidence=report.overall,
                    confidenceThreshold=report.threshold,
                    flaggedFields=report.flaggedFields,
                    rejectedCount=report.rejectedCount,
                    sectionScores=report.sectionScores,
                    timings=timings,
                ),
            )

        if not envelope:
            return assessment.model_dump()

        return ParseResponse(
            assessment=assessment,
            transcript=TranscriptInfo(
                text=transcript.text,
                language=transcript.language,
                durationSeconds=transcript.durationSeconds,
                segments=len(transcript.segments),
                model=transcript.model,
                backend=transcript.backend,
            ),
            confidence=ConfidenceInfo(
                overall=report.overall,
                threshold=report.threshold,
                meetsThreshold=report.meetsThreshold,
                sectionScores=report.sectionScores,
                rejectedCount=report.rejectedCount,
            ),
            flaggedFields=report.flaggedFields,
            timings=timings,
        ).model_dump()
    finally:
        # The temp file must go whether the pipeline succeeded or not.
        try:
            os.unlink(path)
        except OSError:
            logger.warning("Could not remove temp upload %s", path)


@router.post(
    "/assessments",
    response_model=SavedAssessmentResponse,
    status_code=status.HTTP_201_CREATED,
    summary="EP2 - Save a parsed assessment to MongoDB",
    description=(
        "Persist a parsed result. The body is the `assessment` object from "
        "EP1, optionally with the `metadata` that accompanied it.\n\n"
        "The assessment is stored exactly as given - metadata is kept in a "
        "sibling field so the stored assessment stays byte-identical to the "
        "schema the frontend consumes."
    ),
    responses={503: {"model": ErrorResponse, "description": "MongoDB is unavailable"}},
    tags=["assessments"],
)
async def save_assessment(payload: SaveAssessmentRequest):
    try:
        new_id = await repo.save(payload.assessment, payload.metadata)
    except db_client.DatabaseUnavailableError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc

    stored = await repo.get(new_id)
    if stored is None:  # pragma: no cover - only on a concurrent delete
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, "Assessment was saved but could not be read back."
        )

    return SavedAssessmentResponse(
        id=stored.id,
        createdAt=stored.createdAt,
        assessment=stored.assessment,
        metadata=stored.metadata,
    )


@router.get(
    "/assessments",
    response_model=AssessmentListResponse,
    summary="EP4 - List saved assessments, filterable by date",
    description=(
        "List stored assessments, newest first.\n\n"
        "`from` and `to` filter on creation date and accept either a date "
        "(`2026-08-20`) or a full timestamp. A bare `to` date includes that "
        "whole day, so `to=2026-08-20` returns work recorded on the 20th.\n\n"
        "Results are paged; `total` reports how many match the filter."
    ),
    responses={503: {"model": ErrorResponse, "description": "MongoDB is unavailable"}},
    tags=["assessments"],
)
async def list_assessments(
    date_from: datetime | None = Query(
        None, alias="from", description="Only assessments created on or after this date."
    ),
    date_to: datetime | None = Query(
        None, alias="to", description="Only assessments created on or before this date."
    ),
    limit: int = Query(repo.DEFAULT_LIMIT, ge=1, le=repo.MAX_LIMIT),
    skip: int = Query(0, ge=0),
):
    try:
        rows = await repo.list_assessments(
            date_from=date_from, date_to=date_to, limit=limit, skip=skip
        )
        total = await repo.count(date_from=date_from, date_to=date_to)
    except db_client.DatabaseUnavailableError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc

    return AssessmentListResponse(
        total=total,
        count=len(rows),
        limit=limit,
        skip=skip,
        items=[
            SavedAssessmentResponse(
                id=row.id,
                createdAt=row.createdAt,
                assessment=row.assessment,
                metadata=row.metadata,
            )
            for row in rows
        ],
    )


@router.get(
    "/assessments/{assessment_id}",
    response_model=SavedAssessmentResponse,
    summary="EP3 - Retrieve a saved assessment by id",
    description=(
        "Fetch one stored assessment. A malformed id returns 404 rather than "
        "an error, since from the caller's point of view it is the same "
        "situation as an id that does not exist."
    ),
    responses={
        404: {"model": ErrorResponse, "description": "No assessment with that id"},
        503: {"model": ErrorResponse, "description": "MongoDB is unavailable"},
    },
    tags=["assessments"],
)
async def get_assessment(assessment_id: str):
    try:
        stored = await repo.get(assessment_id)
    except db_client.DatabaseUnavailableError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc

    if stored is None:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND, f"No assessment found with id {assessment_id!r}."
        )

    return SavedAssessmentResponse(
        id=stored.id,
        createdAt=stored.createdAt,
        assessment=stored.assessment,
        metadata=stored.metadata,
    )


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Service and dependency health",
    description=(
        "Reports whether MongoDB, the LLM provider and Whisper are reachable. "
        "With three moving parts and no UI, this is the quickest way to "
        "confirm the system is wired up correctly."
    ),
    tags=["system"],
)
async def health():
    settings = get_settings()

    mongo_ok = await db_client.ping()

    llm_ok = False
    try:
        from app.extraction.llm import build_llm

        build_llm(settings)
        llm_ok = True
    except Exception as exc:
        logger.info("LLM health probe failed: %s", exc)

    return HealthResponse(
        status="ok" if mongo_ok else "degraded",
        mongodb=mongo_ok,
        llmProvider=settings.llm_provider,
        llmModel=settings.default_llm_model,
        llmReachable=llm_ok,
        whisperBackend=settings.whisper_backend,
        whisperModel=settings.whisper_model,
        whisperLoaded=get_transcriber()._backend is not None,
    )
