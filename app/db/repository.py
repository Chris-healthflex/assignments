"""Persistence logic for assessments (save / get / list-by-date)."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from app.db.client import db
from app.db.models import from_document, to_document
from app.api.schemas import StoredAssessment


async def save(payload: Dict[str, Any]) -> str:
    """Persist a pipeline/create payload; returns the new id as a string."""
    doc = to_document(payload, db.now())
    res = await db.collection.insert_one(doc)
    return str(res.inserted_id)


async def get(assessment_id: str) -> Optional[StoredAssessment]:
    """Retrieve one assessment by id. Handles both ObjectId and string ids."""
    doc = await _find_by_id(assessment_id)
    return from_document(doc) if doc else None


async def _find_by_id(assessment_id: str) -> Optional[Dict[str, Any]]:
    doc = await db.collection.find_one({"_id": assessment_id})
    if doc is None:
        # real Mongo stores ObjectId; try converting
        try:
            from bson import ObjectId

            doc = await db.collection.find_one({"_id": ObjectId(assessment_id)})
        except Exception:
            doc = None
    return doc


async def list_all(
    start: Optional[datetime] = None, end: Optional[datetime] = None
) -> List[StoredAssessment]:
    """List assessments, newest first, optionally filtered by createdAt range."""
    flt: Dict[str, Any] = {}
    if start or end:
        rng: Dict[str, Any] = {}
        if start:
            rng["$gte"] = start
        if end:
            rng["$lte"] = end
        flt["createdAt"] = rng
    cursor = db.collection.find(flt).sort("createdAt", -1)
    docs = await cursor.to_list(length=1000)
    return [from_document(d) for d in docs]
