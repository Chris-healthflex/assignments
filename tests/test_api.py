import io
from contextlib import asynccontextmanager

import pytest
from fastapi.testclient import TestClient
from mongomock_motor import AsyncMongoMockClient
from pymongo.errors import ServerSelectionTimeoutError

from app.api.assessments import (
    get_extraction_llm,
    get_groq_client,
    get_repository,
    get_transcript_cache,
)
from app.db.mongo import get_repository as build_repository
from app.main import create_app
from app.schemas.first_assessment import ClinicalDetails, FirstAssessment
from app.services.extraction_graph import ExtractionResult, FieldEvidence


_WAV_BYTES = b"RIFF....WAVEfmt "


@asynccontextmanager
async def _noop_lifespan(app):
    yield


class FakeLLM:
    def __init__(self, result: ExtractionResult):
        self._result = result

    def invoke(self, messages):
        return self._result


class FakeGroqClient:
    def __init__(self, transcript_text: str = "Patient reports knee pain."):
        self._transcript_text = transcript_text

    @property
    def audio(self):
        outer = self

        class _Transcriptions:
            def create(self, model, file, response_format=None):
                class _Transcript:
                    text = outer._transcript_text
                    segments = [
                        {"start": 0.0, "end": 3.0, "text": outer._transcript_text}
                    ]

                return _Transcript()

        class _Audio:
            transcriptions = _Transcriptions()

        return _Audio()


@pytest.fixture
def app_client():
    app = create_app(lifespan=_noop_lifespan)

    mongo_client = AsyncMongoMockClient()
    repository = build_repository(mongo_client, "test_db")

    app.dependency_overrides[get_repository] = lambda: repository
    app.dependency_overrides[get_groq_client] = lambda: FakeGroqClient()
    # Disable the shared transcript cache: every test posts byte-identical
    # audio, so a live cache would leak one test's transcript into the next.
    app.dependency_overrides[get_transcript_cache] = lambda: None
    app.dependency_overrides[get_extraction_llm] = lambda: FakeLLM(
        ExtractionResult(
            assessment=FirstAssessment(
                clinicalDetails=ClinicalDetails(chiefComplaint="Knee pain")
            ),
            low_confidence_sections=[],
            evidence=[
                FieldEvidence(
                    field="clinicalDetails.chiefComplaint",
                    segmentIds=[0],
                    quote="Patient reports knee pain.",
                )
            ],
        )
    )

    with TestClient(app) as client:
        yield app, client


def test_parse_returns_structured_assessment(app_client):
    _, client = app_client

    response = client.post(
        "/assessments/parse",
        files={"file": ("session.wav", io.BytesIO(b"RIFF....WAVEfmt "), "audio/wav")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["clinicalDetails"]["chiefComplaint"] == "Knee pain"


def test_parse_rejects_non_wav_file(app_client):
    _, client = app_client

    response = client.post(
        "/assessments/parse",
        files={"file": ("session.mp3", io.BytesIO(b"not a wav"), "audio/mpeg")},
    )

    assert response.status_code == 400


def test_parse_returns_200_when_a_few_sections_are_low_confidence(app_client):
    app, client = app_client
    app.dependency_overrides[get_extraction_llm] = lambda: FakeLLM(
        ExtractionResult(
            assessment=FirstAssessment(),
            low_confidence_sections=["subjectiveGoals", "objectiveGoals"],
        )
    )

    response = client.post(
        "/assessments/parse",
        files={"file": ("session.wav", io.BytesIO(b"RIFF....WAVEfmt "), "audio/wav")},
    )

    assert response.status_code == 200


def test_parse_returns_422_when_broadly_low_confidence(app_client):
    app, client = app_client
    app.dependency_overrides[get_extraction_llm] = lambda: FakeLLM(
        ExtractionResult(
            assessment=FirstAssessment(),
            low_confidence_sections=[
                "subjectiveAssessments",
                "subjectiveGoals",
                "objectiveGoals",
                "recommendation",
                "patientAdvice",
            ],
        )
    )

    response = client.post(
        "/assessments/parse",
        files={"file": ("session.wav", io.BytesIO(b"RIFF....WAVEfmt "), "audio/wav")},
    )

    assert response.status_code == 422
    assert "low_confidence_sections" in response.json()["detail"]


def test_parse_include_debug_wraps_response_and_never_422s(app_client):
    app, client = app_client
    app.dependency_overrides[get_extraction_llm] = lambda: FakeLLM(
        ExtractionResult(
            assessment=FirstAssessment(
                clinicalDetails=ClinicalDetails(chiefComplaint="Knee pain")
            ),
            low_confidence_sections=[
                "subjectiveAssessments",
                "subjectiveGoals",
                "objectiveGoals",
                "recommendation",
                "patientAdvice",
            ],
        )
    )

    response = client.post(
        "/assessments/parse?include_debug=true",
        files={"file": ("session.wav", io.BytesIO(b"RIFF....WAVEfmt "), "audio/wav")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["assessment"]["clinicalDetails"]["chiefComplaint"] == "Knee pain"
    assert body["transcript"] == "Patient reports knee pain."
    assert body["is_low_confidence"] is True
    assert body["low_confidence_sections"] == [
        "subjectiveAssessments",
        "subjectiveGoals",
        "objectiveGoals",
        "recommendation",
        "patientAdvice",
    ]
    assert body["confidence"] == round(2 / 7, 2)


def test_parse_without_include_debug_returns_bare_schema(app_client):
    _, client = app_client

    response = client.post(
        "/assessments/parse",
        files={"file": ("session.wav", io.BytesIO(b"RIFF....WAVEfmt "), "audio/wav")},
    )

    assert response.status_code == 200
    assert set(response.json().keys()) == {
        "clinicalDetails",
        "subjectiveAssessments",
        "objectiveAssessment",
        "subjectiveGoals",
        "objectiveGoals",
        "recommendation",
        "patientAdvice",
    }


def test_parse_include_debug_exposes_transcript_segments_and_evidence(app_client):
    _, client = app_client

    response = client.post(
        "/assessments/parse?include_debug=true",
        files={"file": ("session.wav", io.BytesIO(_WAV_BYTES), "audio/wav")},
    )

    assert response.status_code == 200
    body = response.json()

    assert body["segments"] == [
        {"id": 0, "start": 0.0, "end": 3.0, "text": "Patient reports knee pain."}
    ]
    assert body["evidence"] == [
        {
            "field": "clinicalDetails.chiefComplaint",
            "segmentIds": [0],
            "quote": "Patient reports knee pain.",
        }
    ]
    assert body["ungrounded_fields"] == []
    assert body["validation_issues"] == []
    assert body["attempts"] == 1


def test_parse_include_debug_reports_ungrounded_values(app_client):
    """A value the model cannot cite must be surfaced, not silently trusted."""
    app, client = app_client
    app.dependency_overrides[get_extraction_llm] = lambda: FakeLLM(
        ExtractionResult(
            assessment=FirstAssessment(
                clinicalDetails=ClinicalDetails(
                    chiefComplaint="Knee pain", duration="8 months"
                )
            ),
            evidence=[
                FieldEvidence(field="clinicalDetails.chiefComplaint", segmentIds=[0])
            ],
        )
    )

    response = client.post(
        "/assessments/parse?include_debug=true",
        files={"file": ("session.wav", io.BytesIO(_WAV_BYTES), "audio/wav")},
    )

    body = response.json()
    assert body["ungrounded_fields"] == ["clinicalDetails.duration"]
    assert body["attempts"] == 2  # the graph tried to correct itself


def test_response_carries_a_request_id_header(app_client):
    _, client = app_client

    response = client.get("/assessments")

    assert response.headers.get("X-Request-ID")


def test_create_then_get_assessment(app_client):
    _, client = app_client

    create_response = client.post(
        "/assessments",
        json=FirstAssessment(
            clinicalDetails=ClinicalDetails(chiefComplaint="Shoulder pain")
        ).model_dump(),
    )
    assert create_response.status_code == 201
    assessment_id = create_response.json()["id"]

    get_response = client.get(f"/assessments/{assessment_id}")
    assert get_response.status_code == 200
    assert get_response.json()["clinicalDetails"]["chiefComplaint"] == "Shoulder pain"


def test_get_missing_assessment_returns_404(app_client):
    _, client = app_client

    response = client.get("/assessments/000000000000000000000000")

    assert response.status_code == 404


def test_list_assessments(app_client):
    _, client = app_client
    client.post("/assessments", json=FirstAssessment().model_dump())
    client.post("/assessments", json=FirstAssessment().model_dump())

    response = client.get("/assessments")

    assert response.status_code == 200
    assert len(response.json()) == 2


def test_mongo_unavailable_returns_503(app_client):
    app, client = app_client

    class _BrokenRepository:
        async def list(self, date_from=None, date_to=None):
            raise ServerSelectionTimeoutError("no servers found")

    app.dependency_overrides[get_repository] = lambda: _BrokenRepository()

    response = client.get("/assessments")

    assert response.status_code == 503
    assert response.json() == {"detail": "Database unavailable"}
