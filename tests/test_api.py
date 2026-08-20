"""Tests for the four REST endpoints (D1).

Whisper and the LLM are stubbed and MongoDB is in-memory, so the whole suite
runs in seconds with no models, no GPU and no database server.
"""

from __future__ import annotations

import io
import wave

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.extraction.confidence import ConfidenceReport, FieldFlag
from app.schemas.first_assessment import FirstAssessment, SECTION_KEYS, empty_assessment
from app.transcription.whisper_service import Transcript, TranscriptSegment

TRANSCRIPT_TEXT = (
    "The patient presented with left knee pain following surgery eight months "
    "ago. Left knee flexion was 124 degrees compared with 130 degrees on the "
    "right. Physiotherapy was recommended once weekly for four sessions."
)

GOOD_PAYLOAD = {
    "clinicalDetails": {
        "clinicalHistory": "surgery eight months ago",
        "chiefComplaint": "left knee pain",
        "duration": "eight months",
    },
    "subjectiveAssessments": [{"testName": "Pain", "conclusion": "left knee pain"}],
    "objectiveAssessment": {
        "tests": [{"testName": "Knee flexion", "unitName": "degrees", "left": "124", "right": "130"}]
    },
    "recommendation": [
        {"sessionType": "Physiotherapy", "sessionFrequency": "once weekly for four sessions"}
    ],
}


def wav_bytes(seconds: float = 1.0, rate: int = 16000) -> bytes:
    """A real, decodable WAV, so upload handling is exercised for real."""
    t = np.arange(int(seconds * rate)) / rate
    samples = (0.4 * np.sin(2 * np.pi * 440 * t) * 32767).astype("<i2")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(samples.tobytes())
    return buffer.getvalue()


class StubTranscriber:
    def __init__(self, text: str = TRANSCRIPT_TEXT):
        self.text = text
        self._backend = object()

    def transcribe(self, path):
        return Transcript(
            text=self.text,
            segments=[TranscriptSegment(start=0.0, end=5.0, text=self.text)],
            language="en",
            durationSeconds=105.55,
            backend="faster-whisper",
            model="small",
            transcribeSeconds=24.3,
        )


def stub_extraction(assessment: FirstAssessment, *, confidence: float, flags=None, rejected=0):
    report = ConfidenceReport(
        overall=confidence,
        meetsThreshold=confidence >= 0.55,
        threshold=0.55,
        sectionScores={"clinicalDetails": 1.0},
        flaggedFields=flags or [],
        rejectedCount=rejected,
    )

    def _run(transcript, **kwargs):
        return {
            "assessment": assessment,
            "confidence": report,
            "issues": [],
            "errors": {},
            "timings": {"total": 1.0},
        }

    return _run


@pytest.fixture
def client(mongo, monkeypatch):
    """A TestClient with Whisper and the LLM stubbed and Mongo in memory."""
    from app.api import routes
    from app.main import create_app

    monkeypatch.setattr(routes, "get_transcriber", lambda: StubTranscriber())
    monkeypatch.setattr(
        routes,
        "extract_assessment",
        stub_extraction(FirstAssessment.model_validate(GOOD_PAYLOAD), confidence=0.9),
    )
    with TestClient(create_app()) as test_client:
        yield test_client


# --------------------------------------------------------------------------
# EP1 - POST /assessments/parse
# --------------------------------------------------------------------------
def test_parse_returns_the_assessment_and_metadata(client):
    response = client.post(
        "/assessments/parse", files={"file": ("session.wav", wav_bytes(), "audio/wav")}
    )
    assert response.status_code == 200
    body = response.json()

    assert list(body["assessment"]) == list(SECTION_KEYS)
    assert body["assessment"]["clinicalDetails"]["chiefComplaint"] == "left knee pain"
    assert body["transcript"]["language"] == "en"
    assert body["confidence"]["overall"] == 0.9
    assert "timings" in body


def test_parse_assessment_has_no_extra_keys(client):
    """The frontend contract: exactly seven keys, flags outside."""
    response = client.post(
        "/assessments/parse", files={"file": ("session.wav", wav_bytes(), "audio/wav")}
    )
    assessment = response.json()["assessment"]
    assert set(assessment) == set(SECTION_KEYS)
    assert "confidence" not in assessment
    assert "flaggedFields" not in assessment


def test_parse_with_envelope_false_returns_the_bare_assessment(client):
    response = client.post(
        "/assessments/parse?envelope=false",
        files={"file": ("session.wav", wav_bytes(), "audio/wav")},
    )
    assert response.status_code == 200
    body = response.json()
    assert list(body) == list(SECTION_KEYS)


def test_parse_can_persist_in_one_call(client):
    response = client.post(
        "/assessments/parse?save=true",
        files={"file": ("session.wav", wav_bytes(), "audio/wav")},
    )
    assert response.status_code == 200
    listed = client.get("/assessments").json()
    assert listed["total"] == 1
    assert listed["items"][0]["metadata"]["sourceFilename"] == "session.wav"


def test_parse_reports_flagged_fields(client, monkeypatch):
    from app.api import routes

    flags = [
        FieldFlag(path="objectiveGoals[0].targetDate", reason="not_stated"),
        FieldFlag(path="patientAdvice.adviceDetails", reason="rejected", detail="discarded"),
    ]
    monkeypatch.setattr(
        routes,
        "extract_assessment",
        stub_extraction(
            FirstAssessment.model_validate(GOOD_PAYLOAD), confidence=0.9, flags=flags, rejected=1
        ),
    )
    body = client.post(
        "/assessments/parse", files={"file": ("s.wav", wav_bytes(), "audio/wav")}
    ).json()

    reasons = {flag["reason"] for flag in body["flaggedFields"]}
    assert reasons == {"not_stated", "rejected"}
    assert body["confidence"]["rejectedCount"] == 1


# --------------------------------------------------------------------------
# EP1 error paths
# --------------------------------------------------------------------------
def test_non_wav_upload_is_rejected_with_400(client, monkeypatch):
    from app.api import routes
    from app.transcription.audio_io import InvalidAudioError

    class Failing:
        _backend = None

        def transcribe(self, path):
            raise InvalidAudioError("Not a readable PCM WAV file")

    monkeypatch.setattr(routes, "get_transcriber", lambda: Failing())
    response = client.post(
        "/assessments/parse", files={"file": ("fake.wav", b"ID3 not audio", "audio/wav")}
    )
    assert response.status_code == 400
    assert "PCM WAV" in response.json()["detail"]


def test_empty_upload_is_rejected_with_400(client):
    response = client.post(
        "/assessments/parse", files={"file": ("empty.wav", b"", "audio/wav")}
    )
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_oversized_upload_is_rejected_with_413(client, monkeypatch):
    from app.config import get_settings

    settings = get_settings()
    monkeypatch.setattr(settings, "max_upload_bytes", 1024)
    response = client.post(
        "/assessments/parse", files={"file": ("big.wav", wav_bytes(2.0), "audio/wav")}
    )
    assert response.status_code == 413
    assert "limit" in response.json()["detail"].lower()


def test_low_confidence_returns_422_with_field_level_detail(client, monkeypatch):
    """The brief's explicit requirement."""
    from app.api import routes

    flags = [
        FieldFlag(path="clinicalDetails.chiefComplaint", reason="not_stated"),
        FieldFlag(path="objectiveAssessment.tests", reason="not_stated", detail="no entries"),
    ]
    monkeypatch.setattr(
        routes,
        "extract_assessment",
        stub_extraction(empty_assessment(), confidence=0.12, flags=flags),
    )

    response = client.post(
        "/assessments/parse", files={"file": ("unclear.wav", wav_bytes(), "audio/wav")}
    )
    assert response.status_code == 422

    detail = response.json()["detail"]
    assert detail["confidence"] == 0.12
    assert detail["threshold"] == 0.55
    paths = {field["path"] for field in detail["fields"]}
    assert "clinicalDetails.chiefComplaint" in paths
    # The transcript and partial result are included so a clinician can judge.
    assert detail["transcript"]
    assert list(detail["assessment"]) == list(SECTION_KEYS)


def test_llm_unavailable_returns_503(client, monkeypatch):
    from app.api import routes
    from app.extraction.llm import LLMUnavailableError

    def dead(transcript, **kwargs):
        raise LLMUnavailableError("Ollama is not running on http://localhost:11434")

    monkeypatch.setattr(routes, "extract_assessment", dead)
    response = client.post(
        "/assessments/parse", files={"file": ("s.wav", wav_bytes(), "audio/wav")}
    )
    assert response.status_code == 503
    assert "Ollama" in response.json()["detail"]


# --------------------------------------------------------------------------
# EP2 - POST /assessments
# --------------------------------------------------------------------------
def test_save_returns_201_with_an_id(client):
    response = client.post("/assessments", json={"assessment": GOOD_PAYLOAD})
    assert response.status_code == 201
    body = response.json()
    assert len(body["id"]) == 24
    assert body["assessment"]["clinicalDetails"]["chiefComplaint"] == "left knee pain"


def test_save_stores_metadata_beside_the_assessment(client):
    response = client.post(
        "/assessments",
        json={
            "assessment": GOOD_PAYLOAD,
            "metadata": {"sourceFilename": "clinical_assessment.wav", "confidence": 0.9},
        },
    )
    body = response.json()
    assert body["metadata"]["sourceFilename"] == "clinical_assessment.wav"
    assert set(body["assessment"]) == set(SECTION_KEYS)


def test_save_rejects_an_assessment_with_extra_keys(client):
    """extra="forbid" must be enforced at the API boundary too."""
    response = client.post(
        "/assessments", json={"assessment": {**GOOD_PAYLOAD, "confidence": 0.9}}
    )
    assert response.status_code == 422


# --------------------------------------------------------------------------
# EP3 - GET /assessments/{id}
# --------------------------------------------------------------------------
def test_get_returns_a_saved_assessment(client):
    new_id = client.post("/assessments", json={"assessment": GOOD_PAYLOAD}).json()["id"]
    response = client.get(f"/assessments/{new_id}")

    assert response.status_code == 200
    assert response.json()["assessment"] == FirstAssessment.model_validate(GOOD_PAYLOAD).model_dump()


def test_get_unknown_id_returns_404(client):
    response = client.get("/assessments/507f1f77bcf86cd799439011")
    assert response.status_code == 404


def test_get_malformed_id_returns_404_not_500(client):
    """A client mistake must not read as a server fault."""
    response = client.get("/assessments/not-a-real-id")
    assert response.status_code == 404


# --------------------------------------------------------------------------
# EP4 - GET /assessments
# --------------------------------------------------------------------------
def test_list_is_empty_initially(client):
    body = client.get("/assessments").json()
    assert body == {"total": 0, "count": 0, "limit": 50, "skip": 0, "items": []}


def test_list_returns_saved_assessments(client):
    for _ in range(3):
        client.post("/assessments", json={"assessment": GOOD_PAYLOAD})

    body = client.get("/assessments").json()
    assert body["total"] == 3
    assert len(body["items"]) == 3


def test_list_supports_paging(client):
    for _ in range(5):
        client.post("/assessments", json={"assessment": GOOD_PAYLOAD})

    page = client.get("/assessments?limit=2&skip=2").json()
    assert page["total"] == 5
    assert page["count"] == 2
    assert page["limit"] == 2
    assert page["skip"] == 2


def test_list_filters_by_date(client):
    client.post("/assessments", json={"assessment": GOOD_PAYLOAD})

    assert client.get("/assessments?from=2020-01-01").json()["total"] == 1
    assert client.get("/assessments?to=2020-01-01").json()["total"] == 0


def test_list_rejects_an_out_of_range_limit(client):
    assert client.get("/assessments?limit=0").status_code == 422
    assert client.get("/assessments?limit=99999").status_code == 422


# --------------------------------------------------------------------------
# System
# --------------------------------------------------------------------------
def test_health_reports_dependencies(client):
    body = client.get("/health").json()
    assert body["mongodb"] is True
    assert body["llmProvider"] == "ollama"
    assert "whisperModel" in body


def test_openapi_documents_all_four_endpoints(client):
    paths = client.get("/openapi.json").json()["paths"]
    assert set(paths) == {
        "/assessments/parse",
        "/assessments",
        "/assessments/{assessment_id}",
        "/health",
    }


def test_docs_page_renders(client):
    assert client.get("/docs").status_code == 200


# --------------------------------------------------------------------------
# Clinician interface
# --------------------------------------------------------------------------
def test_index_serves_the_interface(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Clinical Assessment" in response.text


def test_static_assets_serve(client):
    for path in ("/static/styles.css", "/static/app.js"):
        assert client.get(path).status_code == 200, path


def test_interface_is_not_in_the_openapi_schema(client):
    """The UI routes are not API surface and must not clutter the docs."""
    paths = client.get("/openapi.json").json()["paths"]
    assert "/" not in paths
    assert "/static" not in paths


def test_interface_does_not_shadow_the_api(client):
    """Mounting static must not break the endpoints it sits alongside."""
    assert client.get("/health").status_code == 200
    assert client.get("/assessments").status_code == 200


def test_stylesheet_defines_print_rules(client):
    """PDF export is browser print, so the print block is the feature."""
    css = client.get("/static/styles.css").text
    assert "@media print" in css
    assert "@page" in css


def test_responses_carry_tracing_headers(client):
    response = client.get("/health")
    assert response.headers["X-Request-ID"]
    assert float(response.headers["X-Process-Time"]) >= 0


def test_a_recording_with_no_speech_is_400_not_503(client, monkeypatch):
    """The service is fine; the upload is unusable.

    Reporting this as 503 sends the caller hunting for an outage when what
    they need to do is re-record. EmptyTranscriptError subclasses
    TranscriptionError, so the handler order in routes.py is load-bearing.
    """
    from app.api import routes
    from app.transcription.whisper_service import EmptyTranscriptError

    class Silent:
        _backend = None

        def transcribe(self, path):
            raise EmptyTranscriptError("No speech was found in this recording.")

    monkeypatch.setattr(routes, "get_transcriber", lambda: Silent())
    response = client.post(
        "/assessments/parse", files={"file": ("silence.wav", wav_bytes(), "audio/wav")}
    )
    assert response.status_code == 400
    assert "no speech" in response.json()["detail"].lower()


def test_a_broken_whisper_backend_is_still_503(client, monkeypatch):
    """The distinction only holds if the parent case keeps its old status."""
    from app.api import routes
    from app.transcription.whisper_service import TranscriptionError

    class Broken:
        _backend = None

        def transcribe(self, path):
            raise TranscriptionError("faster-whisper is not installed.")

    monkeypatch.setattr(routes, "get_transcriber", lambda: Broken())
    response = client.post(
        "/assessments/parse", files={"file": ("s.wav", wav_bytes(), "audio/wav")}
    )
    assert response.status_code == 503
