"""MongoDB persistence for stored assessments.

Driver note: this uses ``pymongo.AsyncMongoClient`` rather than Motor. Motor is
past end-of-life -- PyMongo absorbed the async API in 4.13 and Motor's final
release was retired in May 2026 -- so starting a new project on it would mean
shipping a dependency with no upstream. The API is near-identical; the only
visible difference is the import.

What goes in a document is deliberately the *envelope*, not the contract. A
``StoredAssessment`` carries the untouched ``FirstAssessment`` under one key and
wraps ids, timestamps, transcript and confidence around it. Nothing in this
module reaches inside ``assessment`` -- if it did, the exact-match guarantee
would depend on the database round-trip preserving it, and that is a promise
better kept by never touching it at all.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, time, timedelta, timezone
from typing import Any

from bson import ObjectId
from bson.errors import InvalidId
from pymongo import DESCENDING, AsyncMongoClient
from pymongo.asynchronous.collection import AsyncCollection
from pymongo.errors import PyMongoError

from app.config import get_settings
from app.schemas import StoredAssessment

logger = logging.getLogger(__name__)

_client: AsyncMongoClient | None = None

# Named so `ensure_indexes` is idempotent and the test can assert on it.
LIST_INDEX = "createdAt_-1__id_-1"


# --------------------------------------------------------------------------- #
# Connection
# --------------------------------------------------------------------------- #
def get_client() -> AsyncMongoClient:
    """Lazily create the process-wide client.

    Constructing the client does not connect; the first operation does. That is
    why the URI being wrong shows up as a timeout on the first save rather than
    at import, and why ``serverSelectionTimeoutMS`` is set short -- an
    unreachable Atlas cluster should fail the request in seconds, not hang the
    worker for the 30s default.
    """
    global _client
    if _client is None:
        settings = get_settings()
        _client = AsyncMongoClient(
            settings.mongodb_uri,
            # Read timestamps back as timezone-aware UTC. Without this, BSON
            # hands back naive datetimes and comparing one to the aware
            # `createdAt` the model produced raises TypeError at runtime.
            tz_aware=True,
            serverSelectionTimeoutMS=settings.mongodb_timeout_ms,
        )
        logger.info("Mongo client created for database %r", settings.mongodb_db)
    return _client


def get_collection() -> AsyncCollection:
    settings = get_settings()
    return get_client()[settings.mongodb_db][settings.mongodb_assessments_collection]


async def ping() -> bool:
    """Cheap liveness check. Never raises -- a health endpoint that 500s is useless."""
    try:
        await get_client().admin.command("ping")
        return True
    except PyMongoError:
        logger.warning("Mongo ping failed", exc_info=True)
        return False


async def ensure_indexes() -> None:
    """Idempotent index creation, called once on startup.

    Compound and descending, matching the list query exactly: the date filter is
    a range on ``createdAt`` and the sort is newest-first with ``_id`` breaking
    ties. Including the tie-break in the index is what lets skip/limit paginate
    without an in-memory sort. Creating it twice is a no-op, so this is safe to
    run on every boot.
    """
    await get_collection().create_index(
        [("createdAt", DESCENDING), ("_id", DESCENDING)], name=LIST_INDEX
    )


async def close() -> None:
    """Close and forget the client, so the next call builds a fresh one."""
    global _client
    if _client is not None:
        await _client.close()
        _client = None


# --------------------------------------------------------------------------- #
# Document conversion
# --------------------------------------------------------------------------- #
def to_document(stored: StoredAssessment) -> dict[str, Any]:
    """Model -> BSON-ready dict.

    ``id`` is dropped: Mongo owns it. Everything else dumps as-is, including
    ``createdAt`` as a real datetime rather than a string, so range queries can
    use it. Note that ``FieldEvidence.confidence`` is a computed property and so
    is *not* written -- only the three raw signals are stored, and the score is
    recomputed on read. Derived values in a database go stale the moment the
    scoring rule changes; the inputs do not.
    """
    doc = stored.model_dump(mode="python")
    doc.pop("id", None)
    return doc


def from_document(doc: dict[str, Any]) -> StoredAssessment:
    """BSON dict -> model, moving ``_id`` into the model's ``id`` field.

    Validation is strict (``extra="forbid"``): a document carrying keys the
    model does not know raises rather than quietly dropping them. That is the
    behaviour we want -- silent divergence between what is stored and what is
    served is the failure mode worth being loud about.
    """
    doc = dict(doc)
    oid = doc.pop("_id", None)
    doc["id"] = str(oid) if oid is not None else ""
    return StoredAssessment.model_validate(doc)


# --------------------------------------------------------------------------- #
# Read / write
# --------------------------------------------------------------------------- #
async def save_assessment(stored: StoredAssessment) -> str:
    """Insert one assessment and return its new id as a string."""
    result = await get_collection().insert_one(to_document(stored))
    assessment_id = str(result.inserted_id)
    logger.info("Saved assessment %s", assessment_id)
    return assessment_id


async def get_assessment(assessment_id: str) -> StoredAssessment | None:
    """Fetch one by id.

    A malformed id returns ``None`` rather than raising: from the caller's side
    "no such assessment" and "that could never be an assessment" both mean 404,
    and making the endpoint tell them apart buys nothing.
    """
    try:
        oid = ObjectId(assessment_id)
    except (InvalidId, TypeError):
        return None
    doc = await get_collection().find_one({"_id": oid})
    return from_document(doc) if doc else None


def _day_range(day: date) -> tuple[datetime, datetime]:
    """Half-open UTC range covering one calendar day.

    Half-open ``[start, next_day)`` rather than ``<= end_of_day``: BSON stores
    milliseconds, so an inclusive upper bound has to pick an end instant, and
    every choice either drops the last fraction of a second or overlaps the next
    day. A half-open range has neither problem.
    """
    start = datetime.combine(day, time.min, tzinfo=timezone.utc)
    return start, start + timedelta(days=1)


async def list_assessments(
    day: date | None = None, *, limit: int = 50, skip: int = 0
) -> list[StoredAssessment]:
    """List assessments, newest first, optionally narrowed to one day.

    The date filters ``createdAt`` -- the envelope's own timestamp. The
    ``FirstAssessment`` contract has no date of its own (the only dates in it are
    the goal ``targetDate`` fields, which are targets, not records of when the
    assessment happened), so the time the assessment was captured is the only
    thing "on this date" can honestly mean here.
    """
    query: dict[str, Any] = {}
    if day is not None:
        start, end = _day_range(day)
        query["createdAt"] = {"$gte": start, "$lt": end}

    cursor = (
        get_collection()
        .find(query)
        .sort([("createdAt", DESCENDING), ("_id", DESCENDING)])
        .skip(skip)
        .limit(limit)
    )
    return [from_document(doc) async for doc in cursor]
