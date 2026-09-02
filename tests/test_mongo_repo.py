"""Integration and unit tests for MongoDB async persistence layer."""

from datetime import datetime, timedelta, timezone
from typing import List
import pytest
import pytest_asyncio

from app.db.models import AssessmentDocument
from app.db.mongo import MongoDBManager
from app.repositories.assessment_repo import AssessmentRepository
from app.schemas.assessment import (
    ClinicalDetails,
    FirstAssessment,
    ObjectiveAssessment,
    ObjectiveTest,
    PatientAdvice,
    Recommendation,
    SubjectiveAssessment,
)


@pytest.fixture
def test_db_manager() -> MongoDBManager:
    """Create a MongoDBManager configured for test execution."""
    return MongoDBManager(db_name="test_clinical_db", collection_name="test_assessments")


@pytest_asyncio.fixture
async def repo(test_db_manager: MongoDBManager):
    """Fixture providing an AssessmentRepository with test collection and automatic cleanup."""
    repository = AssessmentRepository(manager=test_db_manager)
    await repository.ensure_indexes()

    created_ids: List[str] = []

    # Wrapper to track created IDs for teardown cleanup
    original_create = repository.create

    async def tracked_create(*args, **kwargs):
        doc = await original_create(*args, **kwargs)
        created_ids.append(doc.id)
        return doc

    repository.create = tracked_create

    yield repository

    # Cleanup inserted test documents
    for doc_id in created_ids:
        await repository.delete_by_id(doc_id)

    # Close test client connection
    test_db_manager.close()


@pytest.fixture
def sample_assessment() -> FirstAssessment:
    """Create a sample FirstAssessment instance."""
    return FirstAssessment(
        clinicalDetails=ClinicalDetails(
            clinicalHistory="Road traffic accident, tibial condyle fracture, ORIF.",
            chiefComplaint="Left knee pain and walking difficulty.",
            duration={"text": "8 months"},
        ),
        subjectiveAssessments=[
            SubjectiveAssessment(testName="Pain Scale", conclusion=["Moderate pain"])
        ],
        objectiveAssessment=ObjectiveAssessment(
            tests=[
                ObjectiveTest(
                    testName="Knee Flexion",
                    unitName="degrees",
                    left="124",
                    right="130",
                    comments=["Restricted on left"],
                )
            ]
        ),
        recommendation=[
            Recommendation(sessionType="Physiotherapy", sessionFrequency="Once weekly for 4 sessions")
        ],
        patientAdvice=PatientAdvice(
            adviceDetails="Avoid prolonged standing."
        ),
    )


@pytest.mark.asyncio
async def test_mongo_connection_ping(test_db_manager: MongoDBManager):
    """Test 1: MongoDB connection manager pings the local MongoDB server successfully."""
    is_connected = await test_db_manager.ping()
    assert is_connected is True
    test_db_manager.close()


@pytest.mark.asyncio
async def test_create_and_get_assessment(repo: AssessmentRepository, sample_assessment: FirstAssessment):
    """Test 2 & 3: Save a FirstAssessment and retrieve it by ID."""
    doc = await repo.create(sample_assessment, source_audio="clinical_assessment.wav")

    assert doc.id is not None
    assert len(doc.id) == 24  # Standard ObjectId hex string length
    assert doc.source_audio == "clinical_assessment.wav"
    assert doc.assessment.clinicalDetails.chiefComplaint == "Left knee pain and walking difficulty."

    # Retrieve by ID
    retrieved = await repo.get_by_id(doc.id)
    assert retrieved is not None
    assert retrieved.id == doc.id
    assert retrieved.assessment.clinicalDetails.clinicalHistory == sample_assessment.clinicalDetails.clinicalHistory
    assert retrieved.assessment.objectiveAssessment.tests[0].left == "124"


@pytest.mark.asyncio
async def test_get_by_invalid_or_missing_id(repo: AssessmentRepository):
    """Test 4: Missing or invalid ObjectId formats return None cleanly."""
    # Non-existent valid ObjectId
    result = await repo.get_by_id("507f1f77bcf86cd799439011")
    assert result is None

    # Invalid hex format string
    invalid_result = await repo.get_by_id("invalid-id-12345")
    assert invalid_result is None

    # Empty string
    empty_result = await repo.get_by_id("")
    assert empty_result is None


@pytest.mark.asyncio
async def test_list_and_date_filtering(repo: AssessmentRepository, sample_assessment: FirstAssessment):
    """Test 5 & 6: List assessments with date filtering (start_date, end_date)."""
    now = datetime.now(timezone.utc)
    one_hour_ago = now - timedelta(hours=1)
    one_hour_later = now + timedelta(hours=1)

    doc = await repo.create(sample_assessment)

    # 1. Query with wide window -> should find the document
    items, total = await repo.list(start_date=one_hour_ago, end_date=one_hour_later)
    assert total >= 1
    assert any(item.id == doc.id for item in items)

    # 2. Query with past window -> should NOT find this newly created document
    two_hours_ago = now - timedelta(hours=2)
    past_items, past_total = await repo.list(start_date=two_hours_ago, end_date=one_hour_ago)
    assert not any(item.id == doc.id for item in past_items)


@pytest.mark.asyncio
async def test_sorting_and_pagination(repo: AssessmentRepository, sample_assessment: FirstAssessment):
    """Test 7: Pagination (skip/limit) and newest-first sorting."""
    doc1 = await repo.create(sample_assessment)
    doc2 = await repo.create(sample_assessment)

    items, total = await repo.list(limit=1, skip=0)
    assert len(items) == 1
    assert total >= 2

    # Most recent document should be returned first
    assert items[0].id == doc2.id


@pytest.mark.asyncio
async def test_index_creation(repo: AssessmentRepository):
    """Test 8: created_at index exists on the MongoDB collection."""
    index_info = await repo.collection.index_information()
    assert "idx_created_at_desc" in index_info or any("created_at" in str(idx) for idx in index_info.values())


@pytest.mark.asyncio
async def test_first_assessment_schema_isolation(repo: AssessmentRepository, sample_assessment: FirstAssessment):
    """Test 9 & 10: Persistence metadata does not leak into the FirstAssessment production model."""
    doc = await repo.create(sample_assessment)
    retrieved = await repo.get_by_id(doc.id)

    assert retrieved is not None
    # Verify the inner assessment is a pure FirstAssessment
    inner_dump = retrieved.assessment.model_dump()

    # Must contain ONLY the 7 official FirstAssessment keys
    expected_keys = {
        "clinicalDetails",
        "subjectiveAssessments",
        "objectiveAssessment",
        "subjectiveGoals",
        "objectiveGoals",
        "recommendation",
        "patientAdvice",
    }
    assert set(inner_dump.keys()) == expected_keys

    # Must NOT have id or timestamps inside the inner FirstAssessment
    assert "id" not in inner_dump
    assert "_id" not in inner_dump
    assert "created_at" not in inner_dump
    assert "updated_at" not in inner_dump
