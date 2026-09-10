"""FastAPI application exposing the assessment pipeline.

Four endpoints: parse an upload, persist a parsed result, fetch one, and list
by date. Pipeline and repository arrive through ``Depends`` so tests can
substitute them with ``app.dependency_overrides`` rather than through a bespoke
abstraction layer.
"""

import logging
import shutil
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Annotated

import anthropic
from fastapi import Depends, FastAPI, Query, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict
from pymongo import AsyncMongoClient

from .agent import run_agent
from .config import load_settings
from .errors import (
    AssessmentNotFoundError,
    AudioDecodeError,
    ClinicalAssessmentError,
    LowConfidenceError,
    StorageError,
)
from .grounding import FieldVerdict
from .llm import request_extraction
from .schema import FirstAssessment
from .storage import AssessmentRepository, ExtractionSummary, StoredAssessment
from .transcription import transcribe

logger = logging.getLogger(__name__)

UPLOAD_CHUNK_BYTES = 1024 * 1024


@dataclass(frozen=True)
class ParsedAssessment:
    """What the parsing pipeline produces for one recording."""

    assessment: FirstAssessment
    extraction: ExtractionSummary
    transcript: str
    ungrounded_fields: tuple[FieldVerdict, ...]


# Takes a path to a WAV file on disk, returns the parsed assessment.
ParsingPipeline = Callable[[Path], ParsedAssessment]


class ParseResponse(BaseModel):
    """Body of a successful parse: the assessment plus its confidence data."""

    model_config = ConfigDict(extra="forbid")

    assessment: FirstAssessment
    extraction: ExtractionSummary


class SaveAssessmentRequest(BaseModel):
    """Body accepted when persisting a previously parsed assessment."""

    model_config = ConfigDict(extra="forbid")

    assessment: FirstAssessment
    extraction: ExtractionSummary
    transcript: str = ""


class SavedAssessmentResponse(BaseModel):
    """Body returned after persisting an assessment."""

    model_config = ConfigDict(extra="forbid")

    id: str


class StoredAssessmentResponse(BaseModel):
    """Body returned when reading a stored assessment back."""

    model_config = ConfigDict(extra="forbid")

    id: str
    createdAt: datetime
    assessment: FirstAssessment
    extraction: ExtractionSummary


def run_pipeline(wav_path: Path) -> ParsedAssessment:
    """Transcribe a recording and extract a verified assessment from it.

    Args:
        wav_path: Path to a 16-bit PCM WAV file.

    Returns:
        The assessment, its confidence metadata, and the transcript.

    Raises:
        AudioDecodeError: If the file is not readable WAV audio.
        TranscriptionError: If transcription fails.
        ExtractionError: If the model call fails or refuses.
    """
    settings = load_settings()
    transcript = transcribe(wav_path)
    client = anthropic.Anthropic()
    outcome = run_agent(transcript, lambda prompt: request_extraction(client, prompt))
    return ParsedAssessment(
        assessment=outcome.assessment,
        extraction=ExtractionSummary(
            overallConfidence=outcome.confidence,
            lowConfidenceFields=[
                verdict.field_path for verdict in outcome.ungrounded_fields
            ],
            transcriptionModel=settings.whisper_model_size,
            extractionModel=settings.anthropic_model,
        ),
        transcript=transcript,
        ungrounded_fields=outcome.ungrounded_fields,
    )


@lru_cache(maxsize=1)
def _mongo_client() -> AsyncMongoClient:
    """Return the process-wide Mongo client.

    Cached because the driver holds a connection pool: building a client per
    request would leak sockets rather than reuse them.
    """
    return AsyncMongoClient(load_settings().mongodb_uri)


def get_pipeline() -> ParsingPipeline:
    """Provide the parsing pipeline; overridden in tests."""
    return run_pipeline


def get_repository() -> AssessmentRepository:
    """Provide the assessment repository; overridden in tests."""
    return AssessmentRepository(_mongo_client())


app = FastAPI(
    title="Clinical Assessment API",
    description="Transcribe a physiotherapy session and extract an assessment.",
)


@app.post("/assessments/parse", response_model=ParseResponse)
async def parse_assessment(
    file: UploadFile,
    pipeline: Annotated[ParsingPipeline, Depends(get_pipeline)],
) -> ParseResponse:
    """Parse an uploaded WAV recording without persisting anything.

    Args:
        file: Multipart WAV upload.
        pipeline: Injected parsing pipeline.

    Returns:
        The assessment and its confidence metadata.

    Raises:
        LowConfidenceError: If too few fields could be grounded.
    """
    wav_path = await _save_upload(file)
    try:
        # Whisper and the model call are blocking, so they run off the event
        # loop rather than stalling every other request.
        parsed = await run_in_threadpool(pipeline, wav_path)
    finally:
        wav_path.unlink(missing_ok=True)

    threshold = load_settings().confidence_threshold
    if parsed.extraction.overallConfidence < threshold:
        raise LowConfidenceError(
            parsed.extraction.overallConfidence, threshold, parsed.ungrounded_fields
        )
    return ParseResponse(assessment=parsed.assessment, extraction=parsed.extraction)


@app.post("/assessments", response_model=SavedAssessmentResponse, status_code=201)
async def save_assessment(
    body: SaveAssessmentRequest,
    repository: Annotated[AssessmentRepository, Depends(get_repository)],
) -> SavedAssessmentResponse:
    """Persist a parsed assessment.

    Args:
        body: The assessment, its confidence metadata, and the transcript.
        repository: Injected assessment repository.

    Returns:
        The new assessment's id.
    """
    assessment_id = await repository.save(
        body.assessment, body.extraction, body.transcript
    )
    return SavedAssessmentResponse(id=assessment_id)


@app.get("/assessments/{assessment_id}", response_model=StoredAssessmentResponse)
async def get_assessment(
    assessment_id: str,
    repository: Annotated[AssessmentRepository, Depends(get_repository)],
) -> StoredAssessmentResponse:
    """Fetch one stored assessment by id.

    Args:
        assessment_id: The stored assessment's id.
        repository: Injected assessment repository.

    Returns:
        The stored assessment.
    """
    return _to_response(await repository.get_by_id(assessment_id))


@app.get("/assessments", response_model=list[StoredAssessmentResponse])
async def list_assessments(
    repository: Annotated[AssessmentRepository, Depends(get_repository)],
    created_from: Annotated[datetime | None, Query(alias="from")] = None,
    created_to: Annotated[datetime | None, Query(alias="to")] = None,
) -> list[StoredAssessmentResponse]:
    """List stored assessments, optionally filtered by creation date.

    Args:
        repository: Injected assessment repository.
        created_from: Inclusive lower bound, as the ``from`` query parameter.
        created_to: Inclusive upper bound, as the ``to`` query parameter.

    Returns:
        Matching assessments, oldest first.
    """
    stored = await repository.list_by_date_range(created_from, created_to)
    return [_to_response(item) for item in stored]


@app.exception_handler(LowConfidenceError)
async def _handle_low_confidence(
    request: Request, exc: LowConfidenceError
) -> JSONResponse:
    """Return 422 naming each field that could not be verified.

    The ``error`` discriminator is what distinguishes this from FastAPI's own
    422, whose ``detail`` is a list of validation errors rather than an object.
    """
    return JSONResponse(
        status_code=422,
        content={
            "detail": {
                "error": "low_confidence",
                "overallConfidence": exc.confidence,
                "threshold": exc.threshold,
                "lowConfidenceFields": [
                    {
                        "field": verdict.field_path,
                        "reason": verdict.failure.value if verdict.failure else "",
                    }
                    for verdict in exc.ungrounded_fields
                ],
            }
        },
    )


@app.exception_handler(AudioDecodeError)
async def _handle_audio_decode(request: Request, exc: AudioDecodeError) -> JSONResponse:
    """Return 400 for an upload that is not decodable WAV audio."""
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.exception_handler(AssessmentNotFoundError)
async def _handle_not_found(
    request: Request, exc: AssessmentNotFoundError
) -> JSONResponse:
    """Return 404 for an unknown or malformed assessment id."""
    return JSONResponse(status_code=404, content={"detail": str(exc)})


@app.exception_handler(StorageError)
async def _handle_storage_error(request: Request, exc: StorageError) -> JSONResponse:
    """Return 503 when the assessment store is unreachable."""
    logger.error("Storage unavailable: %s", exc)
    return JSONResponse(
        status_code=503, content={"detail": "The assessment store is unavailable"}
    )


@app.exception_handler(ClinicalAssessmentError)
async def _handle_pipeline_error(
    request: Request, exc: ClinicalAssessmentError
) -> JSONResponse:
    """Return 500 for transcription and extraction failures.

    The message is deliberately generic: the underlying error can name a
    temporary file path, which must not reach a client.
    """
    logger.error("Pipeline failure: %s", exc)
    return JSONResponse(
        status_code=500, content={"detail": "Failed to process the recording"}
    )


async def _save_upload(file: UploadFile) -> Path:
    """Stream an upload to a temporary file.

    Copied in chunks rather than read into memory, so a large recording does
    not have to fit in RAM.

    Args:
        file: The multipart upload.

    Returns:
        Path to the temporary file; the caller is responsible for removing it.
    """
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as destination:
        await run_in_threadpool(
            shutil.copyfileobj, file.file, destination, UPLOAD_CHUNK_BYTES
        )
        return Path(destination.name)


def _to_response(stored: StoredAssessment) -> StoredAssessmentResponse:
    """Convert a stored assessment into its response body."""
    return StoredAssessmentResponse(
        id=stored.id,
        createdAt=stored.createdAt,
        assessment=stored.assessment,
        extraction=stored.extraction,
    )
