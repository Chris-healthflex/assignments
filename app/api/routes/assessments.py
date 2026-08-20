from typing import Annotated
from fastapi import APIRouter, Depends, File, Query, UploadFile, status

from app.core.logging import logger
from app.schemas.first_assessment import FirstAssessment
from app.models.assessment import AssessmentDocument, SaveAssessmentResponse, AssessmentListResponse
from app.services.assessment_service import AssessmentService, get_assessment_service

router = APIRouter(prefix="/assessments", tags=["Assessments"])


@router.post(
    "/parse",
    response_model=FirstAssessment,
    status_code=status.HTTP_200_OK,
    summary="EP1 - Parse a single WAV audio into structured FirstAssessment JSON",
    description="Uploads a single combined clinical session WAV recording, transcribes via Whisper, extracts clinical entities via LangGraph, and validates against the FirstAssessment schema."
)
async def parse_assessment(
    file: Annotated[UploadFile, File(description="Clinical session WAV audio recording (combined doctor-patient)")],
    service: Annotated[AssessmentService, Depends(get_assessment_service)],
) -> FirstAssessment:
    logger.info("Received POST /assessments/parse request with file '%s'", file.filename)
    return await service.parse_audio(file)



@router.post(
    "",
    response_model=SaveAssessmentResponse,
    status_code=status.HTTP_201_CREATED,
    summary="EP2 - Save parsed assessment to MongoDB",
    description="Accepts a valid FirstAssessment document and persists it into MongoDB."
)
async def save_assessment(
    assessment: FirstAssessment,
    service: Annotated[AssessmentService, Depends(get_assessment_service)],
) -> SaveAssessmentResponse:
    logger.info("Received POST /assessments save request")
    return await service.save_assessment(assessment)


@router.get(
    "/{id}",
    response_model=AssessmentDocument,
    status_code=status.HTTP_200_OK,
    summary="EP3 - Retrieve assessment by ID",
    description="Retrieves a previously stored assessment document by its unique identifier."
)
async def get_assessment(
    id: str,
    service: Annotated[AssessmentService, Depends(get_assessment_service)],
) -> AssessmentDocument:
    logger.info("Received GET /assessments/%s request", id)
    return await service.get_assessment(id)


@router.get(
    "",
    response_model=AssessmentListResponse,
    status_code=status.HTTP_200_OK,
    summary="EP4 - List stored assessments",

    description="Lists stored assessment documents with optional date filtering (e.g. ?date=2026-08-20)."
)
async def list_assessments(
    service: Annotated[AssessmentService, Depends(get_assessment_service)],
    date: Annotated[str | None, Query(description="Creation date filter in YYYY-MM-DD format")] = None,
) -> AssessmentListResponse:
    logger.info("Received GET /assessments list request (date_filter=%s)", date)
    return await service.list_assessments(date_filter=date)

