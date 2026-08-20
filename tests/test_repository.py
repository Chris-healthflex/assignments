"""Tests for MongoDB persistence (D4).

Run against mongomock, so no server is needed. The same code was additionally
verified against a live MongoDB 7.0.14 during development.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from app.db import repository as repo
from app.db.models import AssessmentMetadata, from_document, to_document
from app.extraction.confidence import FieldFlag
from app.schemas.first_assessment import FirstAssessment, SECTION_KEYS

NOW = datetime(2026, 8, 20, 12, 0, tzinfo=timezone.utc)


# --------------------------------------------------------------------------
# The contract that matters: the stored assessment is unchanged
# --------------------------------------------------------------------------
async def test_round_trip_preserves_the_assessment_exactly(mongo, sample_assessment):
    new_id = await repo.save(sample_assessment)
    stored = await repo.get(new_id)

    assert stored is not None
    assert stored.assessment.model_dump() == sample_assessment.model_dump()


async def test_stored_assessment_keeps_exactly_seven_keys(mongo, sample_assessment):
    new_id = await repo.save(sample_assessment)
    stored = await repo.get(new_id)
    assert list(stored.assessment.model_dump()) == list(SECTION_KEYS)


async def test_metadata_is_stored_beside_the_assessment_not_inside_it(
    mongo, sample_assessment
):
    """Metadata inside the assessment would violate the frontend contract."""
    metadata = AssessmentMetadata(
        sourceFilename="clinical_assessment.wav",
        confidence=0.9,
        rejectedCount=2,
        flaggedFields=[FieldFlag(path="objectiveGoals[0].targetDate", reason="not_stated")],
    )
    new_id = await repo.save(sample_assessment, metadata)
    stored = await repo.get(new_id)

    dumped = stored.assessment.model_dump()
    assert "confidence" not in dumped
    assert "metadata" not in dumped
    assert stored.metadata.confidence == 0.9
    assert stored.metadata.flaggedFields[0].path == "objectiveGoals[0].targetDate"


async def test_id_is_returned_as_a_string(mongo, sample_assessment):
    """ObjectId is not JSON-serialisable, so it must not escape this layer."""
    new_id = await repo.save(sample_assessment)
    assert isinstance(new_id, str)
    assert len(new_id) == 24


async def test_created_at_is_timezone_aware(mongo, sample_assessment):
    """MongoDB returns naive UTC; emitting that would look like local time."""
    new_id = await repo.save(sample_assessment, created_at=NOW)
    stored = await repo.get(new_id)
    assert stored.createdAt.tzinfo is not None


# --------------------------------------------------------------------------
# Retrieval failure modes - these become 404s, never 500s
# --------------------------------------------------------------------------
async def test_missing_id_returns_none(mongo):
    assert await repo.get("507f1f77bcf86cd799439011") is None


@pytest.mark.parametrize("bad_id", ["not-an-objectid", "", "123", "../../etc/passwd"])
async def test_malformed_id_returns_none_rather_than_raising(mongo, bad_id):
    """A client mistake must not surface as a server fault."""
    assert await repo.get(bad_id) is None


# --------------------------------------------------------------------------
# Listing and date filtering (EP4)
# --------------------------------------------------------------------------
async def test_listing_is_newest_first(mongo, sample_assessment):
    for offset in (0, 5, 30):
        await repo.save(sample_assessment, created_at=NOW - timedelta(days=offset))

    rows = await repo.list_assessments()
    dates = [row.createdAt for row in rows]
    assert dates == sorted(dates, reverse=True)


async def test_empty_collection_lists_empty(mongo):
    assert await repo.list_assessments() == []
    assert await repo.count() == 0


async def test_date_from_filters_older_records(mongo, sample_assessment):
    for offset in (0, 5, 30):
        await repo.save(sample_assessment, created_at=NOW - timedelta(days=offset))

    recent = await repo.list_assessments(date_from=NOW - timedelta(days=7))
    assert len(recent) == 2


async def test_date_to_filters_newer_records(mongo, sample_assessment):
    for offset in (0, 5, 30):
        await repo.save(sample_assessment, created_at=NOW - timedelta(days=offset))

    older = await repo.list_assessments(date_to=NOW - timedelta(days=10))
    assert len(older) == 1


async def test_date_range_filters_both_ends(mongo, sample_assessment):
    for offset in (0, 5, 30):
        await repo.save(sample_assessment, created_at=NOW - timedelta(days=offset))

    window = await repo.list_assessments(
        date_from=NOW - timedelta(days=10), date_to=NOW - timedelta(days=1)
    )
    assert len(window) == 1


async def test_bare_date_to_includes_that_whole_day(mongo, sample_assessment):
    """"Up to the 20th" must include the 20th, not stop at midnight.

    An exclusive bound would silently omit that day's assessments, which is
    exactly the day a clinician is most likely to be looking for.
    """
    await repo.save(sample_assessment, created_at=datetime(2026, 8, 20, 16, 30, tzinfo=timezone.utc))

    same_day = await repo.list_assessments(date_to=datetime(2026, 8, 20, tzinfo=timezone.utc))
    assert len(same_day) == 1


async def test_naive_datetimes_are_treated_as_utc(mongo, sample_assessment):
    """Query params arrive without a timezone; assuming local would skew results."""
    await repo.save(sample_assessment, created_at=NOW)
    rows = await repo.list_assessments(date_from=datetime(2026, 8, 19))
    assert len(rows) == 1


async def test_count_matches_the_same_filter(mongo, sample_assessment):
    for offset in (0, 5, 30):
        await repo.save(sample_assessment, created_at=NOW - timedelta(days=offset))

    window = {"date_from": NOW - timedelta(days=7)}
    assert await repo.count(**window) == len(await repo.list_assessments(**window))


# --------------------------------------------------------------------------
# Paging
# --------------------------------------------------------------------------
async def test_limit_and_skip_page_through_results(mongo, sample_assessment):
    for offset in range(5):
        await repo.save(sample_assessment, created_at=NOW - timedelta(days=offset))

    first = await repo.list_assessments(limit=2)
    second = await repo.list_assessments(limit=2, skip=2)

    assert len(first) == 2
    assert len(second) == 2
    assert {row.id for row in first}.isdisjoint({row.id for row in second})


async def test_limit_is_clamped_to_a_maximum(mongo, sample_assessment):
    """An unbounded limit would let one request pull the whole collection."""
    await repo.save(sample_assessment)
    rows = await repo.list_assessments(limit=10_000)
    assert len(rows) <= repo.MAX_LIMIT


async def test_non_positive_limit_still_returns_a_row(mongo, sample_assessment):
    await repo.save(sample_assessment)
    assert len(await repo.list_assessments(limit=0)) == 1


async def test_negative_skip_is_treated_as_zero(mongo, sample_assessment):
    await repo.save(sample_assessment)
    assert len(await repo.list_assessments(skip=-5)) == 1


# --------------------------------------------------------------------------
# Document conversion
# --------------------------------------------------------------------------
def test_to_document_stores_the_assessment_under_its_own_key(sample_assessment):
    document = to_document(sample_assessment)
    assert set(document) == {"createdAt", "assessment", "metadata"}
    assert list(document["assessment"]) == list(SECTION_KEYS)


def test_from_document_tolerates_a_missing_assessment():
    """A partially written document must not crash retrieval."""
    stored = from_document({"_id": "abc", "createdAt": NOW})
    assert stored.assessment == FirstAssessment()
    assert stored.id == "abc"


def test_from_document_makes_naive_timestamps_utc():
    stored = from_document({"_id": "abc", "createdAt": datetime(2026, 8, 20, 12, 0)})
    assert stored.createdAt.tzinfo == timezone.utc


async def test_delete_removes_a_record(mongo, sample_assessment):
    new_id = await repo.save(sample_assessment)
    assert await repo.delete(new_id) is True
    assert await repo.get(new_id) is None


async def test_delete_of_a_malformed_id_is_false_not_an_error(mongo):
    assert await repo.delete("nope") is False
