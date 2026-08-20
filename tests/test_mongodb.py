import uuid
from datetime import datetime, timezone
import mongomock
import pytest

from app.repositories.assessment_repository import AssessmentRepository
from app.schemas.first_assessment import FirstAssessment, ClinicalDetails, PatientAdvice
from app.core.errors import AssessmentNotFoundError


@pytest.fixture
def mock_repo() -> AssessmentRepository:
    client = mongomock.MongoClient()
    db = client["test_clinical_db"]
    return AssessmentRepository(collection=db["assessments"])


@pytest.fixture
def basic_assessment() -> FirstAssessment:
    return FirstAssessment(
        clinicalDetails=ClinicalDetails(
            clinicalHistory="Asthma",
            chiefComplaint="Knee pain",
            duration="2 weeks",
        ),
        subjectiveAssessments=[],
        objectiveAssessment={"tests": []},
        subjectiveGoals=[],
        objectiveGoals=[],
        recommendation=[],
        patientAdvice=PatientAdvice(adviceDetails="Rest and ice"),
    )


def test_mongodb_save_and_retrieve_by_id(mock_repo: AssessmentRepository, basic_assessment: FirstAssessment):
    """Verify saving a FirstAssessment and retrieving it with exact ID."""
    doc = mock_repo.save_assessment(basic_assessment)
    assert doc.id is not None
    assert doc.assessment.clinicalDetails.chiefComplaint == "Knee pain"

    retrieved = mock_repo.get_assessment(doc.id)
    assert retrieved is not None
    assert retrieved.id == doc.id
    assert retrieved.assessment.clinicalDetails.chiefComplaint == "Knee pain"
    assert retrieved.assessment.patientAdvice.adviceDetails == "Rest and ice"


def test_mongodb_retrieve_nonexistent_id_raises_not_found(mock_repo: AssessmentRepository):
    """Verify querying an unknown UUID raises AssessmentNotFoundError."""
    random_id = str(uuid.uuid4())
    with pytest.raises(AssessmentNotFoundError):
        mock_repo.get_assessment(random_id)


def test_mongodb_list_all(mock_repo: AssessmentRepository, basic_assessment: FirstAssessment):
    """Verify listing all saved assessments."""
    for _ in range(3):
        mock_repo.save_assessment(basic_assessment)

    items = mock_repo.list_assessments()
    assert len(items) == 3


def test_mongodb_filter_by_date(mock_repo: AssessmentRepository, basic_assessment: FirstAssessment):
    """Verify date filtering returns records matching target UTC date string."""
    doc = mock_repo.save_assessment(basic_assessment)
    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    items = mock_repo.list_assessments(date_filter=today_str)
    assert len(items) >= 1
    assert any(i.id == doc.id for i in items)


def test_mongodb_filter_by_past_or_future_date_returns_empty(mock_repo: AssessmentRepository, basic_assessment: FirstAssessment):
    """Verify date filtering returns empty list when no records match that date."""
    mock_repo.save_assessment(basic_assessment)
    items = mock_repo.list_assessments(date_filter="2010-01-01")
    assert len(items) == 0
