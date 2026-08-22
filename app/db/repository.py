"""Save and retrieve assessments (D4).

The repository owns every BSON concern - ObjectId conversion, date range
construction, sort order - so the API layer deals only in strings and Pydantic
models.

A malformed id is treated as "not found" rather than an error. A caller who
sends a bad id gets 404, not 500: from the outside those are the same
situation, and raising would turn a client mistake into a server fault.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from bson import ObjectId
from bson.errors import InvalidId

from app.db import client as db_client
from app.db.models import AssessmentMetadata, StoredAssessment, from_document, to_document
from app.schemas.first_assessment import FirstAssessment

logger = logging.getLogger(__name__)

#: Guards against an unbounded listing when a caller omits a limit.
DEFAULT_LIMIT = 50
MAX_LIMIT = 500


def _to_object_id(assessment_id: str) -> ObjectId | None:
    try:
        return ObjectId(assessment_id)
    except (InvalidId, TypeError, ValueError):
        return None


def _as_utc(value: datetime) -> datetime:
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


def _date_query(
    date_from: datetime | None, date_to: datetime | None
) -> dict[str, Any]:
    """Build the createdAt filter shared by listing and counting.

    ``date_to`` is widened to the end of the day when a bare date is given,
    because "everything up to the 20th" is what a clinician means; an
    exclusive bound would silently omit that day's work.
    """
    date_range: dict[str, datetime] = {}
    if date_from is not None:
        date_range["$gte"] = _as_utc(date_from)
    if date_to is not None:
        upper = _as_utc(date_to)
        if (upper.hour, upper.minute, upper.second, upper.microsecond) == (0, 0, 0, 0):
            upper = upper.replace(hour=23, minute=59, second=59, microsecond=999999)
        date_range["$lte"] = upper
    return {"createdAt": date_range} if date_range else {}


async def save(
    assessment: FirstAssessment,
    metadata: AssessmentMetadata | None = None,
    *,
    created_at: datetime | None = None,
) -> str:
    """Insert an assessment and return its id as a string."""
    document = to_document(assessment, metadata, created_at=created_at)
    result = await db_client.get_collection().insert_one(document)
    logger.info("Saved assessment %s", result.inserted_id)
    return str(result.inserted_id)


async def get(assessment_id: str) -> StoredAssessment | None:
    """Fetch one assessment. Returns None for missing *or* malformed ids."""
    object_id = _to_object_id(assessment_id)
    if object_id is None:
        logger.info("Rejected malformed assessment id %r", assessment_id)
        return None

    document = await db_client.get_collection().find_one({"_id": object_id})
    return from_document(document) if document else None


async def list_assessments(
    *,
    date_from: datetime | None = None,
    date_to: datetime | None = None,
    limit: int = DEFAULT_LIMIT,
    skip: int = 0,
) -> list[StoredAssessment]:
    """List assessments newest first, optionally filtered by creation date.

    Newest first, because a clinician opening the list wants today's work.
    """
    query = _date_query(date_from, date_to)

    limit = max(1, min(limit, MAX_LIMIT))

    cursor = (
        db_client.get_collection()
        .find(query)
        .sort("createdAt", -1)
        .skip(max(0, skip))
        .limit(limit)
    )
    return [from_document(document) async for document in cursor]


async def count(
    *, date_from: datetime | None = None, date_to: datetime | None = None
) -> int:
    """Total matching the same filter, so a listing can report what it paged."""
    return await db_client.get_collection().count_documents(
        _date_query(date_from, date_to)
    )


async def delete(assessment_id: str) -> bool:
    """Remove an assessment. Not exposed by the API; used to clean up tests."""
    object_id = _to_object_id(assessment_id)
    if object_id is None:
        return False
    result = await db_client.get_collection().delete_one({"_id": object_id})
    return result.deleted_count > 0
