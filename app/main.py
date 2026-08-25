"""FastAPI entrypoint for the clinical voice assessment service."""
from __future__ import annotations

import shutil
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, HTTPException, Query, UploadFile, status

from .agent import ExtractionError, extract_assessment
from .db import AssessmentRepository, DatabaseError
from .schemas import FirstAssessment, SavedAssessment
from .transcribe import TranscriptionError, transcribe_wav

load_dotenv()
app = FastAPI(title="Clinical Voice Assessment Pipeline", version="1.0.0")


async def get_repository() -> AssessmentRepository:
    try:
        return await AssessmentRepository.from_environment()
    except DatabaseError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/assessments/parse", response_model=FirstAssessment)
async def parse_assessment(file: UploadFile = File(...)) -> FirstAssessment:
    if not file.filename or Path(file.filename).suffix.lower() != ".wav":
        raise HTTPException(status_code=415, detail="Upload a WAV file.")
    with TemporaryDirectory() as directory:
        audio_path = Path(directory) / "session.wav"
        try:
            with audio_path.open("wb") as output:
                shutil.copyfileobj(file.file, output)
            transcript = transcribe_wav(audio_path)
            envelope = extract_assessment(transcript)
        except (TranscriptionError, ExtractionError) as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        finally:
            await file.close()
    if envelope.uncertain_fields:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=[issue.model_dump() for issue in envelope.uncertain_fields],
        )
    return envelope.assessment


@app.post("/assessments", response_model=SavedAssessment, status_code=status.HTTP_201_CREATED)
async def save_assessment(
    assessment: FirstAssessment, repository: AssessmentRepository = Depends(get_repository)
) -> SavedAssessment:
    return await repository.create(assessment)


@app.get("/assessments/{assessment_id}", response_model=SavedAssessment)
async def get_assessment(
    assessment_id: str, repository: AssessmentRepository = Depends(get_repository)
) -> SavedAssessment:
    assessment = await repository.get(assessment_id)
    if assessment is None:
        raise HTTPException(status_code=404, detail="Assessment not found.")
    return assessment


@app.get("/assessments", response_model=list[SavedAssessment])
async def list_assessments(
    created_date: date | None = Query(default=None, alias="date"),
    repository: AssessmentRepository = Depends(get_repository),
) -> list[SavedAssessment]:
    return await repository.list(created_date)
