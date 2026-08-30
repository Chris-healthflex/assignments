from __future__ import annotations

import asyncio
from pathlib import Path

from app.config import Settings
from app.errors import ExtractionConfidenceError
from app.pipeline.extraction_agent import Extractor, extract_assessment
from app.pipeline.transcription import transcribe_wav
from app.schemas.first_assessment import FirstAssessment


async def parse_audio(
    path: Path,
    settings: Settings,
    extractor: Extractor | None = None,
) -> tuple[FirstAssessment, dict[str, float], str]:
    transcript = await asyncio.to_thread(transcribe_wav, path, settings)

    if not transcript.strip():
        raise ExtractionConfidenceError(["transcript"])

    assessment, confidence = await extract_assessment(
        transcript,
        settings,
        extractor=extractor,
    )
    return assessment, confidence, transcript
