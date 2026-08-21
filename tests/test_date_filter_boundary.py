"""Regression test for the to_date inclusive-day boundary bug.

This deliberately does NOT mock list_assessments/save_assessment - it runs the
real query logic against mongomock, so it would have caught the bug found in
manual testing: a record saved mid-day (e.g. 05:50) was excluded when filtering
with to_date set to that same bare date (parsed as midnight).
"""

from datetime import datetime, timedelta

import mongomock
import pytest

from app.db import assessments as assessments_module
from app.models.first_assessment import FirstAssessment


@pytest.fixture
def mock_collection(monkeypatch):
    client = mongomock.MongoClient()
    collection = client["test_db"]["assessments"]
    monkeypatch.setattr(assessments_module, "get_collection", lambda: collection)
    return collection


def test_record_saved_mid_day_is_included_with_same_day_to_date_filter(mock_collection) -> None:
    record = assessments_module.save_assessment(FirstAssessment())

    saved_date = record.created_at.date()
    # Simulates the exact query a user sends when passing a bare date, e.g.
    # "?to_date=2026-08-21" - this parses to midnight, which is BEFORE a record
    # saved later that same day unless the boundary is handled inclusively.
    same_day_midnight = datetime(saved_date.year, saved_date.month, saved_date.day)

    results = assessments_module.list_assessments(to_date=same_day_midnight)

    assert any(r.id == record.id for r in results), (
        "Record saved mid-day was excluded by a to_date filter for the same "
        "calendar day - the to_date boundary must be treated as inclusive of "
        "the whole day, not just midnight."
    )


def test_record_outside_the_day_range_is_excluded(mock_collection) -> None:
    record = assessments_module.save_assessment(FirstAssessment())

    day_before = record.created_at.date() - timedelta(days=1)
    to_date_still_before = datetime(day_before.year, day_before.month, day_before.day)

    results = assessments_module.list_assessments(to_date=to_date_still_before)

    assert all(r.id != record.id for r in results)