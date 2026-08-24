"""Service layer used by the API routes: run pipeline + persist + fetch."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from app.api.schemas import PipelineResult, StoredAssessment
from app.db import repository
from app.services.pipeline import run_from_wav


async def process_and_optionally_save(wav_path: str, save: bool) -> Dict[str, Any]:
    """Run the full pipeline on a WAV; optionally persist the result."""
    result: PipelineResult = run_from_wav(wav_path)
    payload = result.model_dump(mode="json")
    if save:
        new_id = await repository.save(payload)
        payload["id"] = new_id
    return payload


async def save_result(payload: Dict[str, Any]) -> str:
    return await repository.save(payload)


async def get_assessment(assessment_id: str) -> Optional[StoredAssessment]:
    return await repository.get(assessment_id)


async def list_assessments(
    start: Optional[datetime], end: Optional[datetime]
) -> List[StoredAssessment]:
    return await repository.list_all(start, end)
