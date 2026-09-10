"""MongoDB persistence for parsed assessments.

The stored document keeps the exact ``FirstAssessment`` under its own key, with
confidence data as a sibling rather than inside it: "no extra fields" is a
constraint on the assessment object, and the envelope around it is ours.

Uses ``pymongo.AsyncMongoClient``, the supported async driver, rather than
Motor, which reached deprecation in May 2026.
"""

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from bson import ObjectId
from bson.errors import InvalidId
from pydantic import BaseModel, ConfigDict
from pymongo import ASCENDING, AsyncMongoClient
from pymongo.errors import PyMongoError

from .errors import AssessmentNotFoundError, StorageError
from .schema import FirstAssessment

logger = logging.getLogger(__name__)

DEFAULT_DATABASE_NAME = "clinical_assessment"
COLLECTION_NAME = "assessments"


class ExtractionSummary(BaseModel):
    """Confidence metadata stored and returned beside the assessment.

    Attribute names are camelCase because this shape crosses the wire in the
    API response, for the same reason the production schema's are.
    """

    model_config = ConfigDict(extra="forbid")

    overallConfidence: float
    lowConfidenceFields: list[str]
    transcriptionModel: str
    extractionModel: str


@dataclass(frozen=True)
class StoredAssessment:
    """One persisted assessment, as read back out of the store."""

    id: str
    createdAt: datetime
    assessment: FirstAssessment
    extraction: ExtractionSummary
    transcript: str


def normalise_timestamp(moment: datetime) -> datetime:
    """Return a UTC-aware timestamp.

    MongoDB stores datetimes as UTC and hands them back without a timezone, so
    naive values are read as UTC rather than as local time.

    Args:
        moment: An aware or naive datetime.

    Returns:
        The same instant, UTC-aware.
    """
    if moment.tzinfo is None:
        return moment.replace(tzinfo=UTC)
    return moment.astimezone(UTC)


def to_document(
    assessment: FirstAssessment,
    extraction: ExtractionSummary,
    transcript: str,
    created_at: datetime,
) -> dict[str, Any]:
    """Build the document to insert. Pure, so it is testable without a database."""
    return {
        "createdAt": normalise_timestamp(created_at),
        "assessment": assessment.model_dump(),
        "extraction": extraction.model_dump(),
        "transcript": transcript,
    }


def from_document(document: dict[str, Any]) -> StoredAssessment:
    """Rebuild a stored assessment from its document.

    Validating through ``FirstAssessment`` on the way out means a document whose
    keys have drifted fails loudly here rather than reaching the frontend.

    Args:
        document: A raw document from the collection.

    Returns:
        The parsed assessment, with ``_id`` exposed as a string ``id``.
    """
    return StoredAssessment(
        id=str(document["_id"]),
        createdAt=normalise_timestamp(document["createdAt"]),
        assessment=FirstAssessment(**document["assessment"]),
        extraction=ExtractionSummary(**document["extraction"]),
        transcript=document["transcript"],
    )


class AssessmentRepository:
    """Reads and writes assessments in MongoDB."""

    def __init__(
        self,
        client: AsyncMongoClient,
        database_name: str = DEFAULT_DATABASE_NAME,
    ) -> None:
        """Bind the repository to a client and database.

        Args:
            client: An open async client; owned by the caller.
            database_name: Database holding the assessments collection.
        """
        self._collection = client[database_name][COLLECTION_NAME]

    async def ensure_indexes(self) -> None:
        """Create the index the date-range listing needs.

        Raises:
            StorageError: If the store cannot be reached.
        """
        # Ascending on createdAt because EP4 filters and sorts by it.
        await self._run("create the createdAt index", self._create_index())

    async def save(
        self,
        assessment: FirstAssessment,
        extraction: ExtractionSummary,
        transcript: str,
        created_at: datetime | None = None,
    ) -> str:
        """Persist one assessment.

        Args:
            assessment: The assessment to store.
            extraction: Confidence metadata to store beside it.
            transcript: The transcript it was derived from.
            created_at: Creation time; defaults to now, in UTC.

        Returns:
            The new document's id as a string.

        Raises:
            StorageError: If the write fails.
        """
        document = to_document(
            assessment, extraction, transcript, created_at or datetime.now(UTC)
        )
        result = await self._run(
            "save the assessment", self._collection.insert_one(document)
        )
        return str(result.inserted_id)

    async def get_by_id(self, assessment_id: str) -> StoredAssessment:
        """Fetch one assessment by id.

        Args:
            assessment_id: The document id as a string.

        Returns:
            The stored assessment.

        Raises:
            AssessmentNotFoundError: If the id is malformed or matches nothing.
            StorageError: If the read fails.
        """
        object_id = _to_object_id(assessment_id)
        document = await self._run(
            "read the assessment", self._collection.find_one({"_id": object_id})
        )
        if document is None:
            raise AssessmentNotFoundError(f"No assessment with id {assessment_id}")
        return from_document(document)

    async def list_by_date_range(
        self,
        created_from: datetime | None = None,
        created_to: datetime | None = None,
    ) -> list[StoredAssessment]:
        """List assessments, optionally bounded by creation time.

        Args:
            created_from: Inclusive lower bound, or None for unbounded.
            created_to: Inclusive upper bound, or None for unbounded.

        Returns:
            Matching assessments, oldest first.

        Raises:
            StorageError: If the read fails.
        """
        cursor = self._collection.find(_date_range_query(created_from, created_to))
        documents = await self._run(
            "list assessments", cursor.sort("createdAt", ASCENDING).to_list()
        )
        return [from_document(document) for document in documents]

    def _create_index(self) -> Any:
        """Return the index-creation awaitable."""
        return self._collection.create_index([("createdAt", ASCENDING)])

    @staticmethod
    async def _run(action: str, operation: Any) -> Any:
        """Await a driver call, mapping its failures to StorageError.

        Every driver failure - connection refused, timeout, write error - is a
        storage outage from the caller's point of view, so they share one
        translation point rather than repeating try/except at each call site.
        """
        try:
            return await operation
        except PyMongoError as exc:
            logger.error("Failed to %s: %s", action, exc)
            raise StorageError(f"Could not {action}: {exc}") from exc


def _to_object_id(assessment_id: str) -> ObjectId:
    """Parse a document id, treating a malformed one as "not found"."""
    try:
        return ObjectId(assessment_id)
    except (InvalidId, TypeError) as exc:
        # A malformed id cannot match anything, so it is a 404 rather than the
        # 500 an unhandled parse error would produce.
        raise AssessmentNotFoundError(
            f"{assessment_id!r} is not a valid assessment id"
        ) from exc


def _date_range_query(
    created_from: datetime | None, created_to: datetime | None
) -> dict[str, Any]:
    """Build the createdAt filter, omitting bounds that were not given."""
    bounds: dict[str, datetime] = {}
    if created_from is not None:
        bounds["$gte"] = normalise_timestamp(created_from)
    if created_to is not None:
        bounds["$lte"] = normalise_timestamp(created_to)
    return {"createdAt": bounds} if bounds else {}
