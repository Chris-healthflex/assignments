"""Tests for the MongoDB layer.

Two tiers, deliberately:

* Conversion tests run everywhere, with no database. They cover the part that
  can actually corrupt the contract (the model <-> BSON round-trip) and are the
  reason a broken save shows up in CI rather than in a clinician's browser.
* Integration tests need a live Mongo and skip themselves when there is not
  one. Skipping is honest; mocking the driver would only prove that the mock
  behaves like the mock.
"""

from __future__ import annotations

import os
from datetime import date, datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from app import db
from app.config import get_settings
from app.schemas import (
    ClinicalDetails,
    ExtractionFlags,
    FieldEvidence,
    FirstAssessment,
    ObjectiveTest,
    StoredAssessment,
    SubjectiveAssessment,
)

TEST_COLLECTION = "first_assessments_test"

# Cached across the session: without it every integration test pays the full
# server-selection timeout on a machine with no Mongo, and eight 5s skips turn a
# fast suite into a slow one.
_reachable: bool | None = None


def make_stored(**overrides) -> StoredAssessment:
    """A realistic envelope: a populated contract plus its evidence."""
    assessment = FirstAssessment(
        clinicalDetails=ClinicalDetails(
            clinicalHistory="RTA eight months ago, left tibial condyle fracture.",
            chiefComplaint="Pain and stiffness in the left knee",
            duration="eight months",
        ),
        subjectiveAssessments=[
            SubjectiveAssessment(testName="Pain", conclusion="Worse on stairs"),
        ],
    )
    assessment.objectiveAssessment.tests = [
        ObjectiveTest(
            testName="Knee flexion",
            unitName="degrees",
            value="",
            left="124",
            right="130",
            comments="",
        )
    ]
    flags = ExtractionFlags.summarise(
        [
            FieldEvidence(
                field="objectiveAssessment.tests[0].left",
                value="124",
                evidence="left knee flexion 124 degrees",
                evidenceFound=True,
                modelConfidence=0.95,
                audioConfidence=0.52,
                contextConfidence=0.52,
            )
        ]
    )
    payload = {
        "audioFilename": "session.wav",
        "transcript": "left knee flexion 124 degrees",
        "flags": flags,
        "assessment": assessment,
    }
    payload.update(overrides)
    return StoredAssessment(**payload)


# --------------------------------------------------------------------------- #
# Conversion: model <-> document
# --------------------------------------------------------------------------- #
def test_the_document_drops_id_and_lets_mongo_own_it():
    doc = db.to_document(make_stored())
    assert "id" not in doc
    assert "_id" not in doc


def test_created_at_is_stored_as_a_datetime_not_a_string():
    # A string timestamp cannot answer a range query, and the date filter is a
    # range query. This is the assertion that keeps `?date=` working.
    doc = db.to_document(make_stored())
    assert isinstance(doc["createdAt"], datetime)
    assert doc["createdAt"].tzinfo is not None


def test_the_contract_survives_the_round_trip_byte_for_byte():
    stored = make_stored()
    doc = db.to_document(stored)
    doc["_id"] = "507f1f77bcf86cd799439011"

    restored = db.from_document(doc)

    # The whole point of the envelope: what goes in under `assessment` is what
    # comes back out, key for key.
    assert restored.assessment.model_dump() == stored.assessment.model_dump()
    assert restored.id == "507f1f77bcf86cd799439011"
    assert restored.transcript == stored.transcript


def test_confidence_is_recomputed_on_read_not_stored():
    # `confidence` is a property derived from the three raw signals. Storing it
    # would freeze today's scoring rule into every old document.
    stored = make_stored()
    doc = db.to_document(stored)
    assert "confidence" not in doc["flags"]["fields"][0]

    doc["_id"] = "507f1f77bcf86cd799439011"
    restored = db.from_document(doc)
    assert restored.flags.fields[0].confidence == pytest.approx(0.52)


def test_a_document_with_an_unknown_key_is_loud_not_silent():
    doc = db.to_document(make_stored())
    doc["_id"] = "507f1f77bcf86cd799439011"
    doc["smuggledField"] = "should not be here"

    with pytest.raises(ValidationError):
        db.from_document(doc)


def test_a_document_without_an_id_still_loads():
    # `find` with a projection, or a fixture written by hand, may not carry one.
    restored = db.from_document(db.to_document(make_stored()))
    assert restored.id == ""


# --------------------------------------------------------------------------- #
# The date range
# --------------------------------------------------------------------------- #
def test_the_day_range_is_half_open_utc():
    start, end = db._day_range(date(2026, 8, 20))
    assert start == datetime(2026, 8, 20, tzinfo=timezone.utc)
    assert end == datetime(2026, 8, 21, tzinfo=timezone.utc)


def test_the_last_millisecond_of_the_day_is_inside_the_range():
    # An inclusive `<= 23:59:59` upper bound would drop this document. BSON keeps
    # milliseconds, so that boundary is reachable, not theoretical.
    start, end = db._day_range(date(2026, 8, 20))
    last = datetime(2026, 8, 20, 23, 59, 59, 999000, tzinfo=timezone.utc)
    assert start <= last < end


def test_midnight_belongs_to_exactly_one_day():
    _, end_of_20th = db._day_range(date(2026, 8, 20))
    start_of_21st, _ = db._day_range(date(2026, 8, 21))
    midnight = datetime(2026, 8, 21, tzinfo=timezone.utc)
    assert not (midnight < end_of_20th)  # excluded from the 20th
    assert start_of_21st <= midnight  # included in the 21st


# --------------------------------------------------------------------------- #
# Integration: needs a live Mongo
# --------------------------------------------------------------------------- #
@pytest.fixture
async def collection():
    """Point the module at a throwaway collection, or skip.

    `get_settings` is cached, so the override is an env var plus a cache clear.
    The client is closed on the way out because pytest-asyncio gives each test a
    fresh event loop, and a client bound to a closed loop fails on next use.
    """
    global _reachable
    if _reachable is False:
        pytest.skip("no MongoDB reachable at MONGODB_URI")

    previous = os.environ.get("MONGODB_ASSESSMENTS_COLLECTION")
    os.environ["MONGODB_ASSESSMENTS_COLLECTION"] = TEST_COLLECTION
    get_settings.cache_clear()

    _reachable = await db.ping()
    if not _reachable:
        await db.close()
        get_settings.cache_clear()
        pytest.skip("no MongoDB reachable at MONGODB_URI")

    await db.get_collection().delete_many({})
    try:
        yield db.get_collection()
    finally:
        await db.get_collection().drop()
        await db.close()
        if previous is None:
            os.environ.pop("MONGODB_ASSESSMENTS_COLLECTION", None)
        else:
            os.environ["MONGODB_ASSESSMENTS_COLLECTION"] = previous
        get_settings.cache_clear()


async def test_save_then_fetch_returns_the_same_assessment(collection):
    stored = make_stored()
    assessment_id = await db.save_assessment(stored)

    fetched = await db.get_assessment(assessment_id)

    assert fetched is not None
    assert fetched.id == assessment_id
    assert fetched.assessment.model_dump() == stored.assessment.model_dump()
    assert fetched.flags.fields[0].confidence == pytest.approx(0.52)


async def test_a_missing_id_is_none_not_an_exception(collection):
    assert await db.get_assessment("507f1f77bcf86cd799439011") is None


async def test_a_malformed_id_is_none_not_an_exception(collection):
    # Whatever a caller puts in the URL path ends up here. "not-an-objectid"
    # should be a 404, not a 500.
    assert await db.get_assessment("not-an-objectid") is None


async def test_the_date_filter_selects_only_that_day(collection):
    day = date(2026, 8, 20)
    on_day = datetime(2026, 8, 20, 9, 30, tzinfo=timezone.utc)
    await db.save_assessment(make_stored(createdAt=on_day))
    await db.save_assessment(make_stored(createdAt=on_day - timedelta(days=1)))
    await db.save_assessment(make_stored(createdAt=on_day + timedelta(days=1)))

    found = await db.list_assessments(day)

    assert len(found) == 1
    assert found[0].createdAt.date() == day


async def test_the_date_filter_includes_both_edges_of_the_day(collection):
    day = date(2026, 8, 20)
    await db.save_assessment(
        make_stored(createdAt=datetime(2026, 8, 20, 0, 0, 0, tzinfo=timezone.utc))
    )
    await db.save_assessment(
        make_stored(
            createdAt=datetime(2026, 8, 20, 23, 59, 59, 999000, tzinfo=timezone.utc)
        )
    )

    assert len(await db.list_assessments(day)) == 2


async def test_listing_without_a_date_returns_newest_first(collection):
    base = datetime(2026, 8, 20, 9, 0, tzinfo=timezone.utc)
    for offset in (0, 2, 1):
        await db.save_assessment(make_stored(createdAt=base + timedelta(hours=offset)))

    found = await db.list_assessments()

    assert [a.createdAt for a in found] == sorted(
        (a.createdAt for a in found), reverse=True
    )
    assert len(found) == 3


async def test_the_limit_is_respected(collection):
    base = datetime(2026, 8, 20, 9, 0, tzinfo=timezone.utc)
    for offset in range(5):
        await db.save_assessment(make_stored(createdAt=base + timedelta(hours=offset)))

    assert len(await db.list_assessments(limit=2)) == 2


async def test_ensure_indexes_is_idempotent(collection):
    await db.ensure_indexes()
    await db.ensure_indexes()  # a second boot must not fail

    # `list_indexes` is a coroutine returning a cursor, unlike `find`, which
    # returns one directly, so the await is required.
    cursor = await collection.list_indexes()
    names = [idx["name"] async for idx in cursor]
    assert db.LIST_INDEX in names
