import logging
import tempfile
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile
from groq import Groq

from app.config import Settings, get_settings
from app.db.mongo import AssessmentNotFoundError, AssessmentRepository
from app.observability import log_context
from app.schemas.first_assessment import ASSESSMENT_SECTIONS, FirstAssessment
from app.services.extraction_graph import StructuredLLM, run_extraction
from app.services.transcription import (
    TranscriptCache,
    TranscriptionError,
    transcribe_audio,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/assessments", tags=["assessments"])

# Shared across requests so the same recording is never transcribed twice.
_transcript_cache = TranscriptCache()


def get_repository() -> AssessmentRepository:
    # Overridden at app startup (see app.main) to inject the real Motor-backed
    # repository; overridden in tests to inject a mongomock-backed one.
    raise NotImplementedError("Repository dependency not configured")


def get_groq_client(settings: Settings = Depends(get_settings)) -> Groq:
    return Groq(api_key=settings.groq_api_key)


def get_extraction_llm() -> StructuredLLM | None:
    # None tells run_extraction to build the default ChatGroq-backed LLM.
    return None


def get_transcript_cache() -> TranscriptCache | None:
    return _transcript_cache


@router.post("/parse")
async def parse_assessment(
    file: UploadFile,
    include_debug: bool = Query(
        default=False,
        description=(
            "Internal use by the demo frontend only. When true, wraps the "
            "response with the transcript segments, per-field evidence and "
            "validation audit instead of returning the bare FirstAssessment "
            "JSON, and skips the 422 rejection so a human can review and "
            "complete flagged sections before saving."
        ),
    ),
    settings: Settings = Depends(get_settings),
    groq_client: Groq = Depends(get_groq_client),
    llm: StructuredLLM | None = Depends(get_extraction_llm),
    cache: TranscriptCache | None = Depends(get_transcript_cache),
):
    if not file.filename or not file.filename.lower().endswith(".wav"):
        raise HTTPException(status_code=400, detail="Only .wav files are supported")

    with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
        tmp.write(await file.read())
        tmp.flush()

        try:
            transcript = transcribe_audio(
                Path(tmp.name), client=groq_client, cache=cache
            )
        except TranscriptionError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    logger.info(
        "transcribed upload",
        extra=log_context(
            upload=file.filename,
            segments=len(transcript.segments),
            duration_s=round(transcript.duration, 1),
        ),
    )

    report, is_low_confidence = run_extraction(
        transcript,
        llm=llm,
        confidence_threshold=settings.confidence_flag_threshold,
        api_key=settings.groq_api_key,
    )

    logger.info(
        "extraction complete",
        extra=log_context(
            attempts=report.attempts,
            low_confidence_sections=report.low_confidence_sections,
            ungrounded_fields=report.ungrounded_fields,
        ),
    )

    if include_debug:
        flagged = len(report.low_confidence_sections)
        confidence = round(
            (len(ASSESSMENT_SECTIONS) - flagged) / len(ASSESSMENT_SECTIONS), 2
        )
        return {
            "assessment": report.assessment.model_dump(),
            "transcript": transcript.text,
            "segments": [segment.model_dump() for segment in transcript.segments],
            "evidence": [entry.model_dump() for entry in report.evidence],
            "ungrounded_fields": report.ungrounded_fields,
            "validation_issues": report.validation_issues,
            "attempts": report.attempts,
            "is_low_confidence": is_low_confidence,
            "low_confidence_sections": report.low_confidence_sections,
            "confidence": confidence,
        }

    if is_low_confidence:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Extraction confidence below threshold",
                "low_confidence_sections": report.low_confidence_sections,
            },
        )

    # The graded contract: the bare FirstAssessment, no extra keys.
    return report.assessment.model_dump()


@router.post("", status_code=201)
async def create_assessment(
    assessment: FirstAssessment,
    repository: AssessmentRepository = Depends(get_repository),
):
    assessment_id = await repository.save(assessment)
    return {"id": assessment_id}


@router.get("/{assessment_id}")
async def get_assessment(
    assessment_id: str,
    repository: AssessmentRepository = Depends(get_repository),
):
    try:
        return await repository.get(assessment_id)
    except AssessmentNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Assessment not found") from exc


@router.get("")
async def list_assessments(
    date_from: datetime | None = Query(default=None),
    date_to: datetime | None = Query(default=None),
    repository: AssessmentRepository = Depends(get_repository),
):
    return await repository.list(date_from=date_from, date_to=date_to)
