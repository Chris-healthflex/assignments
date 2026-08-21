from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from app.core.config import get_settings
from app.db.assessments import get_assessment, list_assessments, save_assessment
from app.models.api_models import AssessmentRecord
from app.models.first_assessment import FirstAssessment
from app.pipeline.extraction import ClinicalExtractionGraph
from app.pipeline.mapping import low_confidence_fields, map_and_validate
from app.pipeline.transcription import TranscriptionError, WhisperTranscriber

router = APIRouter(prefix="/assessments", tags=["assessments"])


def _parse_transcript(transcript: str) -> FirstAssessment:
    settings = get_settings()
    state = ClinicalExtractionGraph(settings.extraction_model, settings.groq_api_key).extract(transcript)
    issues = low_confidence_fields(state.get("confidence", {}), settings.confidence_threshold)
    if issues:
        raise HTTPException(status_code=422, detail={"message": "Extraction confidence is too low", "fields": issues})
    return map_and_validate(state["assessment"])


@router.post("/parse", response_model=FirstAssessment)
async def parse_assessment(file: UploadFile = File(...)) -> FirstAssessment:
    if not file.filename or Path(file.filename).suffix.lower() != ".wav":
        raise HTTPException(status_code=422, detail="A WAV file is required")
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(suffix=".wav", delete=False) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(await file.read())
        transcript = WhisperTranscriber(get_settings().whisper_model).transcribe(temporary_path)
        return _parse_transcript(transcript)
    except TranscriptionError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    finally:
        if temporary_path:
            temporary_path.unlink(missing_ok=True)


@router.post("", response_model=AssessmentRecord, status_code=status.HTTP_201_CREATED)
def create_assessment(assessment: FirstAssessment) -> AssessmentRecord:
    return save_assessment(map_and_validate(assessment))


@router.get("/{assessment_id}", response_model=AssessmentRecord)
def retrieve_assessment(assessment_id: str) -> AssessmentRecord:
    record = get_assessment(assessment_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Assessment not found")
    return record


@router.get("", response_model=list[AssessmentRecord])
def retrieve_assessments(from_date: str | None = None, to_date: str | None = None) -> list[AssessmentRecord]:
    try:
        parsed_from = datetime.fromisoformat(from_date) if from_date else None
        parsed_to = datetime.fromisoformat(to_date) if to_date else None
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Dates must be valid ISO-8601 values") from exc
    return list_assessments(parsed_from, parsed_to)
