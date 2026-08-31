import logging
import tempfile
from datetime import date, datetime, time, timezone
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, File, Query, Response, UploadFile, status

from app.config import Settings, get_settings
from app.database import AssessmentStore, get_collection
from app.errors import PipelineError
from app.extraction import extract_assessment
from app.models import (
    AssessmentList,
    AssessmentRecord,
    FirstAssessment,
    SaveAssessmentRequest,
)
from app.transcription import transcribe

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/assessments", tags=["assessments"])

WAV_CONTENT_TYPES = {"audio/wav", "audio/x-wav", "audio/wave", "audio/vnd.wave"}


def get_store(settings: Settings = Depends(get_settings)) -> AssessmentStore:
    return AssessmentStore(get_collection(settings))


async def save_upload(upload: UploadFile, max_bytes: int) -> Path:
    filename = (upload.filename or "").lower()
    if not filename.endswith(".wav") and (upload.content_type or "") not in WAV_CONTENT_TYPES:
        raise PipelineError(
            "invalid_audio",
            "Only WAV audio uploads are accepted.",
            400,
            [
                {
                    "field": "file",
                    "message": f"received '{upload.filename}' with content type "
                    f"'{upload.content_type}'",
                }
            ],
        )

    handle = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    size = 0
    try:
        while chunk := await upload.read(1024 * 1024):
            size += len(chunk)
            if size > max_bytes:
                raise PipelineError(
                    "invalid_audio",
                    "The uploaded file is larger than the allowed limit.",
                    400,
                    [{"field": "file", "message": f"maximum size is {max_bytes} bytes"}],
                )
            handle.write(chunk)
    finally:
        handle.close()

    if size == 0:
        Path(handle.name).unlink(missing_ok=True)
        raise PipelineError(
            "invalid_audio",
            "The uploaded file is empty.",
            400,
            [{"field": "file", "message": "no bytes received"}],
        )
    return Path(handle.name)


@router.post("/parse", response_model=FirstAssessment)
async def parse_assessment(
    response: Response,
    file: UploadFile = File(..., description="WAV recording of the assessment"),
    settings: Settings = Depends(get_settings),
) -> FirstAssessment:
    path = await save_upload(file, settings.max_upload_bytes)
    try:
        transcript = await transcribe(path, settings)
        result = await extract_assessment(transcript, settings)
    finally:
        path.unlink(missing_ok=True)

    if result.unextracted_fields:
        logger.info(
            "fields not extracted from %s: %s",
            file.filename,
            ", ".join(result.unextracted_fields),
        )

    # The body stays exactly the FirstAssessment schema, so the tracking of
    # fields that could not be extracted is reported in headers.
    response.headers["X-Extraction-Confidence"] = f"{result.confidence:.2f}"
    response.headers["X-Unextracted-Fields"] = ",".join(result.unextracted_fields)
    return result.assessment


@router.post("", response_model=AssessmentRecord, status_code=status.HTTP_201_CREATED)
async def save_assessment(
    payload: SaveAssessmentRequest, store: AssessmentStore = Depends(get_store)
) -> AssessmentRecord:
    record = await store.save(payload.assessment, payload.metadata)
    return AssessmentRecord.model_validate(record)


@router.get("", response_model=AssessmentList)
async def list_assessments(
    from_date: Optional[date] = Query(
        None, description="Only assessments created on or after this UTC date."
    ),
    to_date: Optional[date] = Query(
        None, description="Only assessments created on or before this UTC date."
    ),
    limit: int = Query(50, ge=1, le=200),
    skip: int = Query(0, ge=0),
    store: AssessmentStore = Depends(get_store),
) -> AssessmentList:
    created_from = (
        datetime.combine(from_date, time.min, tzinfo=timezone.utc) if from_date else None
    )
    created_to = (
        datetime.combine(to_date, time.max, tzinfo=timezone.utc) if to_date else None
    )
    records = await store.list(created_from, created_to, limit, skip)
    return AssessmentList(
        count=len(records),
        items=[AssessmentRecord.model_validate(record) for record in records],
    )


@router.get("/{assessment_id}", response_model=AssessmentRecord)
async def get_assessment(
    assessment_id: str, store: AssessmentStore = Depends(get_store)
) -> AssessmentRecord:
    record = await store.get(assessment_id)
    return AssessmentRecord.model_validate(record)
