"""Serialization helpers between API models and Mongo documents."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict

from app.api.schemas import StoredAssessment


def to_document(payload: Dict[str, Any], created_at: datetime) -> Dict[str, Any]:
    """Build the Mongo document from a validated pipeline/create payload."""
    return {
        "createdAt": created_at,
        "assessment": payload.get("assessment", {}),
        "transcript": payload.get("transcript"),
        "confidence": payload.get("confidence"),
        "flaggedFields": payload.get("flaggedFields", []),
        "timings": payload.get("timings", {}),
    }


def from_document(doc: Dict[str, Any]) -> StoredAssessment:
    """Rehydrate a StoredAssessment from a Mongo document."""
    return StoredAssessment(
        id=str(doc["_id"]),
        createdAt=doc["createdAt"],
        assessment=doc["assessment"],
        transcript=doc.get("transcript"),
        confidence=doc.get("confidence"),
        flaggedFields=doc.get("flaggedFields", []),
        timings=doc.get("timings", {}),
    )
