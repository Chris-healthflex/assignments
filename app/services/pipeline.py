from __future__ import annotations

from pathlib import Path

import shutil

from app.models.assessment import FirstAssessment
from app.services.confidence import validate_confidence
from app.services.extraction import extract_assessment
from app.services.transcription import transcribe_audio


class AssessmentPipelineError(Exception):
    pass


def process_audio(
    audio_path: str | Path,
) -> tuple[FirstAssessment, str]:

    path = Path(audio_path)

    if not path.exists():
        raise AssessmentPipelineError(
            f"Audio file does not exist: {path}"
        )

    if not path.is_file():
        raise AssessmentPipelineError(
            f"Audio path is not a file: {path}"
        )

    try:

        transcript = transcribe_audio(
            path
        )

        extraction = extract_assessment(
            transcript
        )

        validate_confidence(
            extraction.confidence,
            assessment=extraction.assessment,
        )

        return (
            extraction.assessment,
            transcript,
        )

    except AssessmentPipelineError:
        raise

    except Exception as exc:
        raise AssessmentPipelineError(
            str(exc)
        ) from exc


def save_uploaded_file(
    source_file,
    destination_directory: str | Path,
    filename: str,
) -> Path:

    destination = Path(
        destination_directory
    )

    destination.mkdir(
        parents=True,
        exist_ok=True,
    )

    safe_name = Path(filename).name

    destination_file = (
        destination / safe_name
    )

    with destination_file.open("wb") as output:
        shutil.copyfileobj(
            source_file,
            output,
        )

    return destination_file
