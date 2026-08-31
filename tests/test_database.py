from datetime import datetime, timedelta, timezone

import pytest
from bson import ObjectId

from app.database import AssessmentStore, close_client, get_collection
from app.errors import PipelineError
from app.models import FirstAssessment

ASSESSMENT = FirstAssessment.model_validate(
    {
        "clinicalDetails": {"chiefComplaint": "Left knee pain", "duration": "8 months"},
        "objectiveAssessment": {
            "tests": [{"testName": "Knee flexion", "unitName": "degrees", "left": "124"}]
        },
    }
)


@pytest.fixture
async def store(settings):
    collection = get_collection(settings)
    try:
        await collection.database.client.admin.command("ping")
    except Exception:
        pytest.skip("MongoDB is not running on MONGODB_URI")
    await collection.delete_many({})
    yield AssessmentStore(collection)
    await collection.delete_many({})
    close_client()


async def test_an_assessment_survives_a_save_and_get_unchanged(store):
    saved = await store.save(ASSESSMENT, {"sourceFile": "clinical_assessment.wav"})

    fetched = await store.get(saved["id"])

    assert fetched["assessment"] == ASSESSMENT.model_dump()
    assert fetched["metadata"]["sourceFile"] == "clinical_assessment.wav"


async def test_listing_filters_by_created_date(store):
    old = await store.save(ASSESSMENT)
    recent = await store.save(ASSESSMENT)
    cutoff = datetime(2020, 1, 1, tzinfo=timezone.utc)
    await store.collection.update_one(
        {"_id": ObjectId(old["id"])}, {"$set": {"createdAt": cutoff}}
    )

    since_yesterday = await store.list(
        created_from=datetime.now(timezone.utc) - timedelta(days=1)
    )
    archived = await store.list(created_to=cutoff + timedelta(days=1))

    assert [record["id"] for record in since_yesterday] == [recent["id"]]
    assert [record["id"] for record in archived] == [old["id"]]


async def test_an_unknown_or_malformed_id_is_reported_as_not_found(store):
    with pytest.raises(PipelineError) as unknown:
        await store.get("000000000000000000000042")
    with pytest.raises(PipelineError) as malformed:
        await store.get("not-an-object-id")

    assert unknown.value.status_code == 404
    assert malformed.value.details[0]["field"] == "id"
