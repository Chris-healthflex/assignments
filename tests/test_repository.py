"""Repository save / get / list against the in-memory backend."""
import pytest
from datetime import datetime, timezone, timedelta
from app.db.client import db
from app.db import repository


@pytest.fixture(autouse=True)
async def _connect():
    await db.connect()
    yield
    await db.close()


@pytest.mark.asyncio
async def test_save_and_get():
    payload = {"assessment": {"clinicalDetails": {"duration": "eight months"}},
               "flaggedFields": [], "timings": {}}
    new_id = await repository.save(payload)
    stored = await repository.get(new_id)
    assert stored is not None
    assert stored.assessment.clinicalDetails.duration == "eight months"


@pytest.mark.asyncio
async def test_list_and_date_filter():
    await repository.save({"assessment": {}, "flaggedFields": [], "timings": {}})
    all_items = await repository.list_all()
    assert len(all_items) >= 1
    future = datetime.now(timezone.utc) + timedelta(days=1)
    none_after = await repository.list_all(start=future)
    assert none_after == []
