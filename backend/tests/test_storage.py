"""Tests for the assessment repository.

Serialisation is unit-tested with no database at all; only the round trip needs
live MongoDB, and that test skips cleanly when MONGODB_URI is unset:

    docker run -d -p 27017:27017 --name stance-mongo mongo:7
"""

import os
from datetime import UTC, datetime, timedelta, timezone
from typing import Any

import pytest
from bson import ObjectId
from pymongo.errors import ConnectionFailure

from clinical_assessment.errors import AssessmentNotFoundError, StorageError
from clinical_assessment.schema import (
    ClinicalDetails,
    FirstAssessment,
    ObjectiveAssessment,
    PatientAdvice,
)
from clinical_assessment.storage import (
    AssessmentRepository,
    ExtractionSummary,
    from_document,
    normalise_timestamp,
    to_document,
)

MONGODB_URI = os.environ.get("MONGODB_URI")
CREATED_AT = datetime(2026, 9, 10, 12, 30, tzinfo=UTC)


def build_assessment() -> FirstAssessment:
    """Build a minimal valid assessment to store."""
    return FirstAssessment(
        clinicalDetails=ClinicalDetails(
            clinicalHistory="",
            chiefComplaint="Right shoulder pain",
            duration="three weeks",
        ),
        subjectiveAssessments=[],
        objectiveAssessment=ObjectiveAssessment(tests=[]),
        subjectiveGoals=[],
        objectiveGoals=[],
        recommendation=[],
        patientAdvice=PatientAdvice(adviceDetails=""),
    )


def build_summary() -> ExtractionSummary:
    """Build extraction metadata to store beside the assessment."""
    return ExtractionSummary(
        overallConfidence=0.95,
        lowConfidenceFields=["objectiveAssessment.tests[0].value"],
        transcriptionModel="small",
        extractionModel="claude-opus-5",
    )


class StubCollection:
    """Stands in for an async Mongo collection."""

    def __init__(
        self, document: dict[str, Any] | None = None, error: Exception | None = None
    ) -> None:
        self._document = document
        self._error = error
        self.queries: list[dict[str, Any]] = []

    async def find_one(self, query: dict[str, Any]) -> dict[str, Any] | None:
        self.queries.append(query)
        if self._error is not None:
            raise self._error
        return self._document


class StubDatabase:
    """Returns the one stub collection regardless of name."""

    def __init__(self, collection: StubCollection) -> None:
        self._collection = collection

    def __getitem__(self, name: str) -> StubCollection:
        return self._collection


class StubClient:
    """Minimal stand-in for AsyncMongoClient's indexing behaviour."""

    def __init__(self, collection: StubCollection) -> None:
        self._database = StubDatabase(collection)

    def __getitem__(self, name: str) -> StubDatabase:
        return self._database


def build_repository(
    document: dict[str, Any] | None = None, error: Exception | None = None
) -> AssessmentRepository:
    """Build a repository backed by a stub collection."""
    collection = StubCollection(document, error)
    return AssessmentRepository(StubClient(collection))  # type: ignore[arg-type]


def test_document_keeps_the_assessment_under_its_own_key() -> None:
    document = to_document(
        build_assessment(), build_summary(), "transcript", CREATED_AT
    )

    assert document["assessment"]["clinicalDetails"]["chiefComplaint"] == (
        "Right shoulder pain"
    )


def test_confidence_data_sits_beside_the_assessment_never_inside_it() -> None:
    document = to_document(
        build_assessment(), build_summary(), "transcript", CREATED_AT
    )

    assert "overallConfidence" not in document["assessment"]


def test_document_serialisation_maps_objectid_to_string_id() -> None:
    object_id = ObjectId()
    document = to_document(build_assessment(), build_summary(), "t", CREATED_AT)
    document["_id"] = object_id

    stored = from_document(document)

    assert stored.id == str(object_id)


def test_round_trip_through_document_preserves_the_assessment() -> None:
    assessment = build_assessment()
    document = to_document(assessment, build_summary(), "t", CREATED_AT)
    document["_id"] = ObjectId()

    stored = from_document(document)

    assert stored.assessment == assessment


def test_naive_datetime_is_normalised_to_utc() -> None:
    naive = datetime(2026, 9, 10, 12, 30)

    normalised = normalise_timestamp(naive)

    assert normalised == datetime(2026, 9, 10, 12, 30, tzinfo=UTC)


def test_aware_datetime_is_converted_to_utc() -> None:
    two_hours_ahead = datetime(2026, 9, 10, 14, 30, tzinfo=timezone(timedelta(hours=2)))

    normalised = normalise_timestamp(two_hours_ahead)

    assert normalised == datetime(2026, 9, 10, 12, 30, tzinfo=UTC)


@pytest.mark.asyncio
async def test_malformed_object_id_raises_not_found() -> None:
    repository = build_repository()

    with pytest.raises(AssessmentNotFoundError, match="not a valid assessment id"):
        await repository.get_by_id("definitely-not-an-object-id")


@pytest.mark.asyncio
async def test_malformed_id_never_reaches_the_database() -> None:
    collection = StubCollection()
    repository = AssessmentRepository(StubClient(collection))  # type: ignore[arg-type]

    with pytest.raises(AssessmentNotFoundError):
        await repository.get_by_id("nope")

    assert collection.queries == []


@pytest.mark.asyncio
async def test_unknown_id_raises_not_found() -> None:
    repository = build_repository(document=None)

    with pytest.raises(AssessmentNotFoundError, match="No assessment with id"):
        await repository.get_by_id(str(ObjectId()))


@pytest.mark.asyncio
async def test_driver_failure_is_wrapped_in_storage_error() -> None:
    repository = build_repository(error=ConnectionFailure("connection refused"))

    with pytest.raises(StorageError, match="Could not read the assessment"):
        await repository.get_by_id(str(ObjectId()))


@pytest.mark.integration
@pytest.mark.skipif(MONGODB_URI is None, reason="MONGODB_URI is not set")
@pytest.mark.asyncio
async def test_save_then_get_round_trips() -> None:
    from pymongo import AsyncMongoClient

    # Fail in seconds rather than sitting on the 30s default when no server is up.
    client: AsyncMongoClient = AsyncMongoClient(
        MONGODB_URI, serverSelectionTimeoutMS=5000
    )
    repository = AssessmentRepository(client, database_name="clinical_assessment_test")
    assessment = build_assessment()
    try:
        await repository.ensure_indexes()
        assessment_id = await repository.save(assessment, build_summary(), "transcript")

        stored = await repository.get_by_id(assessment_id)

        assert stored.assessment == assessment
    finally:
        await client.drop_database("clinical_assessment_test")
        await client.close()


@pytest.mark.integration
@pytest.mark.skipif(MONGODB_URI is None, reason="MONGODB_URI is not set")
@pytest.mark.asyncio
async def test_list_filters_by_date_range() -> None:
    from pymongo import AsyncMongoClient

    # Fail in seconds rather than sitting on the 30s default when no server is up.
    client: AsyncMongoClient = AsyncMongoClient(
        MONGODB_URI, serverSelectionTimeoutMS=5000
    )
    repository = AssessmentRepository(client, database_name="clinical_assessment_test")
    try:
        await repository.save(
            build_assessment(), build_summary(), "old", datetime(2026, 1, 1, tzinfo=UTC)
        )
        await repository.save(
            build_assessment(), build_summary(), "new", datetime(2026, 9, 1, tzinfo=UTC)
        )

        listed = await repository.list_by_date_range(
            created_from=datetime(2026, 6, 1, tzinfo=UTC)
        )

        assert [stored.transcript for stored in listed] == ["new"]
    finally:
        await client.drop_database("clinical_assessment_test")
        await client.close()
