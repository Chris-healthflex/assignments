from __future__ import annotations

import logging
import tempfile
from datetime import date
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from pydantic import ValidationError

from app.config import get_settings
from app.db.mongo import get_assessment_by_id, list_assessments, save_assessment
from app.errors import BadRequestError, ExtractionConfidenceError, TranscriptionError
from app.pipeline.pipeline import parse_audio
from app.schemas.first_assessment import FirstAssessment

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/assessments", tags=["assessments"])

WAV_CONTENT_TYPES = {"audio/wav", "audio/x-wav", "audio/wave", "audio/vnd.wave"}


async def save_upload_to_temp_wav(file: UploadFile) -> Path:
    if file.content_type and file.content_type not in WAV_CONTENT_TYPES:
        raise BadRequestError("File must be a WAV recording")

    header = await file.read(12)
    if not header:
        raise BadRequestError("Audio file is empty")
    if len(header) < 12 or not (header.startswith(b"RIFF") and header[8:12] == b"WAVE"):
        raise BadRequestError("File must be a valid WAV recording")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp:
        temp.write(header)
        while chunk := await file.read(1024 * 1024):
            temp.write(chunk)
        return Path(temp.name)


@router.post("/parse", response_model=FirstAssessment)
async def parse_assessment(file: UploadFile = File(...)) -> FirstAssessment:
    temp_path: Path | None = None

    try:
        temp_path = await save_upload_to_temp_wav(file)
        assessment, _, _ = await parse_audio(temp_path, get_settings())
        return assessment
    except BadRequestError as exc:
        raise HTTPException(status_code=400, detail=exc.message) from exc
    except ExtractionConfidenceError as exc:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Extraction confidence is too low",
                "fields": exc.low_confidence_fields,
            },
        ) from exc
    except TranscriptionError as exc:
        raise HTTPException(status_code=400, detail=exc.message) from exc
    except ValidationError as exc:
        logger.info("Extracted assessment failed schema validation")
        raise HTTPException(status_code=422, detail="Extracted assessment is malformed") from exc
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


@router.post("", status_code=201)
async def create_assessment(assessment: FirstAssessment):
    return await save_assessment(assessment)


@router.get("/{assessment_id}")
async def read_assessment(assessment_id: str):
    try:
        assessment = await get_assessment_by_id(assessment_id)
    except BadRequestError as exc:
        raise HTTPException(status_code=400, detail=exc.message) from exc

    if assessment is None:
        raise HTTPException(status_code=404, detail="Assessment not found")

    return assessment


@router.get("")
async def read_assessments(created_date: str | None = Query(default=None, alias="date")):
    parsed_date: date | None = None
    if created_date is not None:
        try:
            parsed_date = date.fromisoformat(created_date)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="date must use YYYY-MM-DD") from exc

    return await list_assessments(parsed_date)
