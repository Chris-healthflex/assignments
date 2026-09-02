from __future__ import annotations

import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import (
    FastAPI,
    File,
    HTTPException,
    Query,
    UploadFile,
)
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.database import close_mongo_client
from app.models.assessment import FirstAssessment
from app.repositories.assessment_repository import (
    AssessmentRepository,
)
from app.services.confidence import (
    ExtractionConfidenceError,
)
from app.services.pipeline import (
    AssessmentPipelineError,
    process_audio,
)


settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "Clinical audio to structured FirstAssessment JSON."
    ),
)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": settings.app_name,
    }


@app.post(
    "/assessments/parse",
    response_model=FirstAssessment,
    response_model_exclude_none=True,
)
async def parse_assessment(
    file: UploadFile = File(...),
):
    """
    Upload WAV -> Whisper -> LangGraph -> FirstAssessment.
    """

    filename = file.filename or ""

    if not filename.lower().endswith(".wav"):
        raise HTTPException(
            status_code=422,
            detail=[
                {
                    "loc": ["file"],
                    "msg": "Only WAV files are accepted.",
                    "type": "value_error.wav",
                }
            ],
        )

    max_size = (
        settings.max_audio_size_mb
        * 1024
        * 1024
    )

    temporary_path: Optional[Path] = None

    try:
        with tempfile.NamedTemporaryFile(
            suffix=".wav",
            delete=False,
        ) as temp:
            temporary_path = Path(temp.name)

            total_bytes = 0

            while True:
                chunk = await file.read(1024 * 1024)

                if not chunk:
                    break

                total_bytes += len(chunk)

                if total_bytes > max_size:
                    raise HTTPException(
                        status_code=422,
                        detail=[
                            {
                                "loc": ["file"],
                                "msg": (
                                    "Audio file exceeds the maximum "
                                    f"size of {settings.max_audio_size_mb} MB."
                                ),
                                "type": "value_error.file_size",
                            }
                        ],
                    )

                temp.write(chunk)

        assessment, _transcript = process_audio(
            temporary_path
        )

        return assessment

    except HTTPException:
        raise

    except ExtractionConfidenceError as exc:
        return JSONResponse(
            status_code=422,
            content={
                "detail": [
                    item.model_dump()
                    for item in exc.details
                ]
            },
        )

    except AssessmentPipelineError as exc:
        raise HTTPException(
            status_code=422,
            detail=[
                {
                    "loc": ["file"],
                    "msg": str(exc),
                    "type": "value_error.extraction",
                }
            ],
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                "Unexpected assessment processing error: "
                f"{exc}"
            ),
        ) from exc

    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink(
                    missing_ok=True
                )
            except Exception:
                pass

        await file.close()


@app.post(
    "/assessments",
    response_model=dict,
)
def create_assessment(
    assessment: FirstAssessment,
):
    """
    Persist a previously parsed FirstAssessment.
    """

    try:
        repository = AssessmentRepository()

        assessment_id = repository.create(
            assessment
        )

        return {
            "id": assessment_id,
            "assessment": assessment.model_dump(
                mode="json"
            ),
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                "Failed to save assessment: "
                f"{exc}"
            ),
        ) from exc


@app.get(
    "/assessments/{assessment_id}",
    response_model=dict,
)
def get_assessment(
    assessment_id: str,
):
    """
    Retrieve an assessment by MongoDB ObjectId.
    """

    try:
        repository = AssessmentRepository()

        result = repository.get_by_id(
            assessment_id
        )

        if result is None:
            raise HTTPException(
                status_code=404,
                detail="Assessment not found.",
            )

        return result

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                "Failed to retrieve assessment: "
                f"{exc}"
            ),
        ) from exc


@app.get(
    "/assessments",
    response_model=list[dict],
)
def list_assessments(
    from_date: Optional[datetime] = Query(
        default=None,
        description=(
            "Return records created on/after this ISO datetime."
        ),
    ),
    to_date: Optional[datetime] = Query(
        default=None,
        description=(
            "Return records created on/before this ISO datetime."
        ),
    ),
):
    """
    List assessments, optionally filtered by creation date.
    """

    if (
        from_date is not None
        and to_date is not None
        and from_date > to_date
    ):
        raise HTTPException(
            status_code=422,
            detail=[
                {
                    "loc": ["query"],
                    "msg": (
                        "from_date must be earlier than "
                        "or equal to to_date."
                    ),
                    "type": "value_error.date_range",
                }
            ],
        )

    try:
        repository = AssessmentRepository()

        return repository.list(
            from_date=from_date,
            to_date=to_date,
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                "Failed to list assessments: "
                f"{exc}"
            ),
        ) from exc


@app.on_event("shutdown")
def shutdown_event():
    close_mongo_client()
