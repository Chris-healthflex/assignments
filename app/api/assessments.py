import tempfile
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile
from openai import OpenAI

from app.config import Settings, get_settings
from app.db.mongo import AssessmentNotFoundError, AssessmentRepository
from app.schemas.first_assessment import FirstAssessment
from app.services.extraction_graph import StructuredLLM, run_extraction
from app.services.transcription import TranscriptionError, transcribe_audio

router = APIRouter(prefix="/assessments", tags=["assessments"])


def get_repository() -> AssessmentRepository:
    # Overridden at app startup (see app.main) to inject the real Motor-backed
    # repository; overridden in tests to inject a mongomock-backed one.
    raise NotImplementedError("Repository dependency not configured")


def get_openai_client(settings: Settings = Depends(get_settings)) -> OpenAI:
    return OpenAI(api_key=settings.openai_api_key)


def get_extraction_llm() -> StructuredLLM | None:
    # None tells run_extraction to build the default ChatOpenAI-backed LLM.
    return None


@router.post("/parse")
async def parse_assessment(
    file: UploadFile,
    settings: Settings = Depends(get_settings),
    openai_client: OpenAI = Depends(get_openai_client),
    llm: StructuredLLM | None = Depends(get_extraction_llm),
):
    if not file.filename or not file.filename.lower().endswith(".wav"):
        raise HTTPException(status_code=400, detail="Only .wav files are supported")

    with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
        tmp.write(await file.read())
        tmp.flush()

        try:
            transcript = transcribe_audio(Path(tmp.name), client=openai_client)
        except TranscriptionError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    result, is_low_confidence = run_extraction(
        transcript, llm=llm, confidence_threshold=settings.confidence_flag_threshold
    )

    if is_low_confidence:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Extraction confidence below threshold",
                "low_confidence_sections": result.low_confidence_sections,
            },
        )

    return result.assessment.model_dump()


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
