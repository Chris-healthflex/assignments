from __future__ import annotations
import logging
import shutil
import tempfile
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse

from app.agents.extraction import run_extraction
from app.agents.llm import ConfigurationError
from app.config import Settings, get_settings
from app.db.repository import (
    AssessmentRepository,
    InvalidAssessmentId,
    StorageError,
)
from app.schemas.api import (
    AssessmentListResponse,
    ExtractionMeta,
    HealthResponse,
    LowConfidenceDetail,
    ParseResponse,
    SaveAssessmentRequest,
    StoredAssessment,
)
from app.schemas.assessment import FirstAssessment
from app.services.transcription import TranscriptionError, transcribe

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)-8s %(name)s: %(message)s"
)
logger = logging.getLogger("app")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    repo = AssessmentRepository(settings)
    await repo.connect()
    app.state.repository = repo
    app.state.graph = None
    logger.info(
        "Ready. whisper=%s/%s llm=%s/%s db=%s",
        settings.whisper_backend, settings.whisper_model,
        settings.llm_provider, settings.llm_model, settings.mongodb_db,
    )
    try:
        yield
    finally:
        await repo.close()


app = FastAPI(
    title="Clinical Assessment Pipeline",
    version="2.0.0",
    description=(
        "Transcribes a clinician-patient session and extracts a structured "
        "FirstAssessment record."
    ),
    lifespan=lifespan,
)


def get_repository(request: Request) -> AssessmentRepository:
    return request.app.state.repository


SettingsDep = Annotated[Settings, Depends(get_settings)]
RepoDep = Annotated[AssessmentRepository, Depends(get_repository)]

@app.get("/health", response_model=HealthResponse, tags=["ops"])
async def health(repo: RepoDep, settings: SettingsDep) -> HealthResponse:
    mongo_ok = await repo.ping()
    return HealthResponse(
        status="ok" if mongo_ok else "degraded",
        mongo="up" if mongo_ok else "down",
        whisperBackend=settings.whisper_backend,
        llmProvider=settings.llm_provider,
    )

@app.post(
    "/assessments/parse",
    response_model=ParseResponse,
    response_model_exclude_none=False,
    tags=["pipeline"],
    summary="Transcribe a WAV session and extract a FirstAssessment",
)
async def parse_assessment(
    request: Request,
    settings: SettingsDep,
    file: Annotated[UploadFile, File(description="Mono/stereo PCM WAV recording")],
    bare: Annotated[
        bool,
        Query(description="Return only the FirstAssessment object, without meta."),
    ] = False,
    save: Annotated[
        bool, Query(description="Also persist the result to MongoDB.")
    ] = False,
):
    filename = file.filename or "upload.wav"
    if not filename.lower().endswith(".wav"):
        raise HTTPException(
            status_code=415,
            detail=f"Only .wav uploads are accepted (received '{filename}').",
        )

    tmp_dir = Path(tempfile.mkdtemp(prefix="assessment-"))
    tmp_path = tmp_dir / "audio.wav"
    try:
        written = 0
        with tmp_path.open("wb") as out:
            while chunk := await file.read(1024 * 1024):
                written += len(chunk)
                if written > settings.max_upload_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=(
                            f"Upload exceeds the {settings.max_upload_bytes} byte limit."
                        ),
                    )
                out.write(chunk)

        if written == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        try:
            transcript = await run_in_threadpool(transcribe, tmp_path, settings)
        except TranscriptionError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        if request.app.state.graph is None:
            from app.agents.extraction import build_graph

            try:
                request.app.state.graph = build_graph(settings)
            except ConfigurationError as exc:
                raise HTTPException(status_code=503, detail=str(exc)) from exc

        outcome = await run_in_threadpool(
            run_extraction, transcript.text, settings, request.app.state.graph
        )
    finally:
        await file.close()
        shutil.rmtree(tmp_dir, ignore_errors=True)

    meta = outcome.meta.model_copy(
        update={
            "sourceFilename": filename,
            "transcriptLanguage": transcript.language,
            "audioDurationSeconds": round(transcript.duration, 2),
        }
    )

    if meta.overallConfidence < settings.confidence_threshold:
        detail = LowConfidenceDetail(
            message=(
                "Extraction confidence is below the configured threshold; "
                "clinician review is required before this record can be used."
            ),
            overallConfidence=meta.overallConfidence,
            threshold=settings.confidence_threshold,
            fields=meta.fieldConfidence,
            unextractedFields=meta.unextractedFields,
        )
        return JSONResponse(status_code=422, content=detail.model_dump(mode="json"))

    if save:
        repo: AssessmentRepository = request.app.state.repository
        try:
            await repo.save(outcome.assessment, meta)
        except StorageError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    if bare:
        return JSONResponse(content=outcome.assessment.model_dump(mode="json"))
    return ParseResponse(assessment=outcome.assessment, meta=meta)

@app.post(
    "/assessments",
    response_model=StoredAssessment,
    status_code=201,
    tags=["storage"],
    summary="Persist a parsed assessment",
)
async def create_assessment(
    payload: SaveAssessmentRequest, repo: RepoDep
) -> StoredAssessment:
    try:
        return await repo.save(payload.assessment, payload.meta)
    except StorageError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

@app.get(
    "/assessments",
    response_model=AssessmentListResponse,
    tags=["storage"],
    summary="List assessments, optionally filtered by creation date",
)
async def list_assessments(
    repo: RepoDep,
    date_from: Annotated[
        datetime | None, Query(alias="from", description="Inclusive ISO-8601 lower bound")
    ] = None,
    date_to: Annotated[
        datetime | None, Query(alias="to", description="Inclusive ISO-8601 upper bound")
    ] = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    skip: Annotated[int, Query(ge=0)] = 0,
) -> AssessmentListResponse:
    for name, value in (("from", date_from), ("to", date_to)):
        if value is not None and value.tzinfo is None:
            if name == "from":
                date_from = value.replace(tzinfo=timezone.utc)
            else:
                date_to = value.replace(tzinfo=timezone.utc)

    if date_from and date_to and date_from > date_to:
        raise HTTPException(status_code=400, detail="'from' must not be after 'to'.")

    try:
        total, items = await repo.list(date_from, date_to, limit, skip)
    except StorageError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return AssessmentListResponse(total=total, count=len(items), items=items)

@app.get(
    "/assessments/{assessment_id}",
    response_model=StoredAssessment,
    tags=["storage"],
    summary="Retrieve one assessment by id",
)
async def get_assessment(assessment_id: str, repo: RepoDep) -> StoredAssessment:
    try:
        stored = await repo.get(assessment_id)
    except InvalidAssessmentId:
        raise HTTPException(
            status_code=400, detail=f"'{assessment_id}' is not a valid assessment id."
        )
    except StorageError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    if stored is None:
        raise HTTPException(status_code=404, detail="Assessment not found.")
    return stored
