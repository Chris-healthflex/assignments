from datetime import datetime, timedelta, timezone

import pytest
from mongomock_motor import AsyncMongoMockClient

from app.db.mongo import AssessmentNotFoundError, get_repository
from app.schemas.first_assessment import ClinicalDetails, FirstAssessment


@pytest.fixture
def repository():
    client = AsyncMongoMockClient()
    return get_repository(client, "test_db")


@pytest.mark.asyncio
async def test_save_and_get_round_trip(repository):
    assessment = FirstAssessment(
        clinicalDetails=ClinicalDetails(chiefComplaint="Knee pain")
    )

    assessment_id = await repository.save(assessment)
    fetched = await repository.get(assessment_id)

    assert fetched["id"] == assessment_id
    assert fetched["clinicalDetails"]["chiefComplaint"] == "Knee pain"
    assert "createdAt" in fetched


@pytest.mark.asyncio
async def test_get_missing_id_raises(repository):
    with pytest.raises(AssessmentNotFoundError):
        await repository.get("not-a-valid-object-id")


@pytest.mark.asyncio
async def test_list_filters_by_date_range(repository):
    old_id = await repository.save(FirstAssessment())
    await repository._collection.update_one(
        {"_id": __import__("bson").ObjectId(old_id)},
        {"$set": {"createdAt": datetime.now(timezone.utc) - timedelta(days=10)}},
    )
    await repository.save(FirstAssessment())

    recent_only = await repository.list(
        date_from=datetime.now(timezone.utc) - timedelta(days=1)
    )

    assert len(recent_only) == 1
    assert all(r["id"] != old_id for r in recent_only)


@pytest.mark.asyncio
async def test_list_returns_all_when_no_filter(repository):
    await repository.save(FirstAssessment())
    await repository.save(FirstAssessment())

    results = await repository.list()

    assert len(results) == 2
