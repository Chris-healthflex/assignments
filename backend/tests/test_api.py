"""Tests for the four API endpoints.

Both the pipeline and the repository are replaced through
``app.dependency_overrides``: no Whisper, no model call, no MongoDB.
"""

from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from test_schema import PRODUCTION_KEY_PATHS, flatten_key_paths

from clinical_assessment.api import (
    ParsedAssessment,
    app,
    get_pipeline,
    get_repository,
)
from clinical_assessment.errors import (
    AssessmentNotFoundError,
    AudioDecodeError,
    StorageError,
    TranscriptionError,
)
from clinical_assessment.grounding import FieldVerdict, GroundingFailure
from clinical_assessment.schema import (
    ClinicalDetails,
    FirstAssessment,
    ObjectiveAssessment,
    ObjectiveGoal,
    ObjectiveTest,
    PatientAdvice,
    Recommendation,
    SubjectiveAssessment,
    SubjectiveGoal,
)
from clinical_assessment.storage import ExtractionSummary, StoredAssessment

UNVERIFIED = FieldVerdict(
    field_path="objectiveAssessment.tests[0].value",
    value="45",
    evidence="abduction was measured",
    failure=GroundingFailure.DIGITS_NOT_IN_EVIDENCE,
)


def build_assessment() -> FirstAssessment:
    """Build a valid assessment with one entry in every array.

    Every list is populated because the key-path conformance test compares the
    response against the full production key set, and an empty array carries
    none of its item's paths.
    """
    return FirstAssessment(
        clinicalDetails=ClinicalDetails(
            clinicalHistory="",
            chiefComplaint="Right shoulder pain",
            duration="three weeks",
        ),
        subjectiveAssessments=[
            SubjectiveAssessment(testName="Pain on reach", conclusion="Limited")
        ],
        objectiveAssessment=ObjectiveAssessment(
            tests=[
                ObjectiveTest(
                    testName="Abduction",
                    unitName="degrees",
                    value="120",
                    left="",
                    right="120",
                    comments="",
                )
            ]
        ),
        subjectiveGoals=[
            SubjectiveGoal(goalDetails="Sleep on the shoulder", targetDate="")
        ],
        objectiveGoals=[
            ObjectiveGoal(
                goalName="Restore abduction",
                goalCategory="Range of motion",
                unitName="degrees",
                value="160",
                targetDate="",
            )
        ],
        recommendation=[
            Recommendation(sessionType="Physiotherapy", sessionFrequency="Twice weekly")
        ],
        patientAdvice=PatientAdvice(adviceDetails=""),
    )


def build_summary(confidence: float = 1.0) -> ExtractionSummary:
    """Build extraction metadata at a given confidence."""
    return ExtractionSummary(
        overallConfidence=confidence,
        lowConfidenceFields=[] if confidence == 1.0 else [UNVERIFIED.field_path],
        transcriptionModel="small",
        extractionModel="claude-opus-5",
    )


def build_parsed(confidence: float = 1.0) -> ParsedAssessment:
    """Build a pipeline result at a given confidence."""
    return ParsedAssessment(
        assessment=build_assessment(),
        extraction=build_summary(confidence),
        transcript="Abduction is 120 degrees on the right.",
        ungrounded_fields=() if confidence == 1.0 else (UNVERIFIED,),
    )


class FakeRepository:
    """In-memory stand-in for AssessmentRepository."""

    def __init__(self, error: Exception | None = None) -> None:
        self._stored: dict[str, StoredAssessment] = {}
        self._error = error
        self._next_id = 1

    async def save(
        self,
        assessment: FirstAssessment,
        extraction: ExtractionSummary,
        transcript: str,
        created_at: datetime | None = None,
    ) -> str:
        if self._error is not None:
            raise self._error
        assessment_id = f"{self._next_id:024d}"
        self._next_id += 1
        self._stored[assessment_id] = StoredAssessment(
            id=assessment_id,
            createdAt=created_at or datetime.now(UTC),
            assessment=assessment,
            extraction=extraction,
            transcript=transcript,
        )
        return assessment_id

    async def get_by_id(self, assessment_id: str) -> StoredAssessment:
        if self._error is not None:
            raise self._error
        if assessment_id not in self._stored:
            raise AssessmentNotFoundError(f"No assessment with id {assessment_id}")
        return self._stored[assessment_id]

    async def list_by_date_range(
        self,
        created_from: datetime | None = None,
        created_to: datetime | None = None,
    ) -> list[StoredAssessment]:
        if self._error is not None:
            raise self._error
        return [
            item
            for item in sorted(self._stored.values(), key=lambda s: s.createdAt)
            if (created_from is None or item.createdAt >= created_from)
            and (created_to is None or item.createdAt <= created_to)
        ]


@pytest.fixture
def repository() -> FakeRepository:
    """A fresh in-memory repository per test."""
    return FakeRepository()


@pytest.fixture
def client(repository: FakeRepository) -> Iterator[TestClient]:
    """A TestClient with a fully grounded pipeline and the fake repository."""
    app.dependency_overrides[get_pipeline] = lambda: lambda path: build_parsed()
    app.dependency_overrides[get_repository] = lambda: repository
    yield TestClient(app)
    app.dependency_overrides.clear()


def upload() -> dict[str, tuple[str, bytes, str]]:
    """Build the multipart payload for an upload."""
    return {"file": ("session.wav", b"RIFF....WAVEfmt ", "audio/wav")}


def override_pipeline(pipeline: object) -> None:
    """Point the parse endpoint at a specific pipeline callable."""
    app.dependency_overrides[get_pipeline] = lambda: pipeline


def test_parse_returns_exact_assessment_key_set(client: TestClient) -> None:
    response = client.post("/assessments/parse", files=upload())

    assert flatten_key_paths(response.json()["assessment"]) == set(PRODUCTION_KEY_PATHS)


def test_parse_returns_confidence_beside_the_assessment(client: TestClient) -> None:
    response = client.post("/assessments/parse", files=upload())

    assert response.json()["extraction"]["overallConfidence"] == 1.0


def test_parse_does_not_persist(client: TestClient, repository: FakeRepository) -> None:
    client.post("/assessments/parse", files=upload())

    assert repository._stored == {}


def test_confidence_exactly_at_threshold_returns_200(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CONFIDENCE_THRESHOLD", "0.7")
    override_pipeline(lambda path: build_parsed(confidence=0.7))

    response = client.post("/assessments/parse", files=upload())

    assert response.status_code == 200


def test_low_confidence_returns_422_with_field_level_detail(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CONFIDENCE_THRESHOLD", "0.7")
    override_pipeline(lambda path: build_parsed(confidence=0.5))

    response = client.post("/assessments/parse", files=upload())

    assert response.json()["detail"]["lowConfidenceFields"] == [
        {
            "field": "objectiveAssessment.tests[0].value",
            "reason": GroundingFailure.DIGITS_NOT_IN_EVIDENCE.value,
        }
    ]


def test_low_confidence_body_is_distinguishable_from_validation_422(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CONFIDENCE_THRESHOLD", "0.7")
    override_pipeline(lambda path: build_parsed(confidence=0.5))

    response = client.post("/assessments/parse", files=upload())

    assert (response.status_code, response.json()["detail"]["error"]) == (
        422,
        "low_confidence",
    )


def test_missing_upload_returns_fastapis_own_422(client: TestClient) -> None:
    response = client.post("/assessments/parse")

    assert response.status_code == 422 and isinstance(response.json()["detail"], list)


def test_non_wav_upload_returns_400(client: TestClient) -> None:
    def undecodable(path: Path) -> ParsedAssessment:
        raise AudioDecodeError("session.wav is not readable WAV audio")

    override_pipeline(undecodable)

    response = client.post("/assessments/parse", files=upload())

    assert response.status_code == 400


def test_transcription_failure_returns_500(client: TestClient) -> None:
    def failing(path: Path) -> ParsedAssessment:
        raise TranscriptionError("whisper fell over")

    override_pipeline(failing)

    response = client.post("/assessments/parse", files=upload())

    assert response.status_code == 500


def test_pipeline_failure_does_not_leak_internal_detail(client: TestClient) -> None:
    def failing(path: Path) -> ParsedAssessment:
        raise TranscriptionError(r"whisper failed on C:\Users\secret\session.wav")

    override_pipeline(failing)

    response = client.post("/assessments/parse", files=upload())

    assert "secret" not in response.text


def test_save_returns_the_new_id(client: TestClient) -> None:
    response = client.post("/assessments", json=_save_body())

    assert response.status_code == 201 and response.json()["id"]


def test_save_then_get_returns_identical_assessment(client: TestClient) -> None:
    saved = client.post("/assessments", json=_save_body())

    fetched = client.get(f"/assessments/{saved.json()['id']}")

    assert fetched.json()["assessment"] == _save_body()["assessment"]


def test_unknown_id_returns_404(client: TestClient) -> None:
    response = client.get("/assessments/000000000000000000000099")

    assert response.status_code == 404


def test_storage_outage_returns_503(repository: FakeRepository) -> None:
    app.dependency_overrides[get_repository] = lambda: FakeRepository(
        error=StorageError("connection refused")
    )
    try:
        response = TestClient(app).post("/assessments", json=_save_body())
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 503


def test_storage_outage_hides_the_connection_string(repository: FakeRepository) -> None:
    app.dependency_overrides[get_repository] = lambda: FakeRepository(
        error=StorageError("could not reach mongodb://user:pw@localhost:27017")
    )
    try:
        response = TestClient(app).post("/assessments", json=_save_body())
    finally:
        app.dependency_overrides.clear()

    assert "mongodb://" not in response.text


def test_list_returns_saved_assessments(client: TestClient) -> None:
    client.post("/assessments", json=_save_body())

    response = client.get("/assessments")

    assert len(response.json()) == 1


def test_list_filters_by_date_range(
    client: TestClient, repository: FakeRepository
) -> None:
    import asyncio

    asyncio.run(
        repository.save(
            build_assessment(),
            build_summary(),
            "old",
            datetime(2026, 1, 1, tzinfo=UTC),
        )
    )
    asyncio.run(
        repository.save(
            build_assessment(),
            build_summary(),
            "new",
            datetime(2026, 9, 1, tzinfo=UTC),
        )
    )

    response = client.get("/assessments", params={"from": "2026-06-01T00:00:00Z"})

    assert [item["createdAt"] for item in response.json()] == ["2026-09-01T00:00:00Z"]


def _save_body() -> dict[str, object]:
    """Build the JSON body accepted by POST /assessments."""
    return {
        "assessment": build_assessment().model_dump(),
        "extraction": build_summary().model_dump(),
        "transcript": "Abduction is 120 degrees on the right.",
    }
