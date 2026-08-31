import wave
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import pytest
from httpx import ASGITransport, AsyncClient

from app.errors import PipelineError
from app.main import app
from app.models import FirstAssessment
from app.routes import get_store
from app.config import get_settings
from tests.conftest import TRANSCRIPT, StubLLM
from tests.test_schema import SECTIONS


class FakeStore:
    def __init__(self, fail: Optional[PipelineError] = None):
        self.records: List[Dict[str, Any]] = []
        self.fail = fail

    async def save(self, assessment: FirstAssessment, metadata=None) -> Dict[str, Any]:
        if self.fail:
            raise self.fail
        record = {
            "id": f"{len(self.records) + 1:024d}",
            "createdAt": datetime.now(timezone.utc),
            "assessment": assessment.model_dump(),
            "metadata": metadata or {},
        }
        self.records.append(record)
        return record

    async def get(self, assessment_id: str) -> Dict[str, Any]:
        for record in self.records:
            if record["id"] == assessment_id:
                return record
        raise PipelineError(
            "assessment_not_found",
            "Assessment was not found.",
            404,
            [{"field": "id", "message": "no assessment with this id"}],
        )

    async def list(self, created_from=None, created_to=None, limit=50, skip=0):
        return [
            record
            for record in self.records
            if (created_from is None or record["createdAt"] >= created_from)
            and (created_to is None or record["createdAt"] <= created_to)
        ][skip : skip + limit]


@pytest.fixture
def wav_bytes(tmp_path) -> bytes:
    path = tmp_path / "note.wav"
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(np.zeros(16000, dtype=np.int16).tobytes())
    return path.read_bytes()


@pytest.fixture
def store() -> FakeStore:
    return FakeStore()


@pytest.fixture
def client(settings, store, monkeypatch):
    monkeypatch.setattr("app.routes.transcribe", _stub_transcribe(TRANSCRIPT))
    monkeypatch.setattr("app.routes.extract_assessment", _stub_extract(StubLLM()))
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_store] = lambda: store
    yield AsyncClient(transport=ASGITransport(app=app), base_url="http://test")
    app.dependency_overrides.clear()


def _stub_transcribe(text: str, error: Optional[PipelineError] = None):
    async def transcribe(path, settings):
        if error:
            raise error
        return text

    return transcribe


def _stub_extract(llm):
    from app.extraction import extract_assessment as real_extract

    async def extract(transcript, settings):
        return await real_extract(transcript, settings, llm)

    return extract


async def test_parse_returns_only_the_first_assessment_schema(client, wav_bytes):
    async with client as http:
        response = await http.post(
            "/assessments/parse",
            files={"file": ("clinical_assessment.wav", wav_bytes, "audio/wav")},
        )

    assert response.status_code == 200
    body = response.json()
    assert set(body) == SECTIONS
    assert FirstAssessment.model_validate(body).model_dump() == body
    assert response.headers["x-extraction-confidence"] == "0.90"
    assert "patientAdvice.adviceDetails" in response.headers["x-unextracted-fields"]


async def test_parse_rejects_anything_that_is_not_a_wav(client):
    async with client as http:
        response = await http.post(
            "/assessments/parse", files={"file": ("note.mp3", b"not-wav", "audio/mpeg")}
        )

    assert response.status_code == 400
    body = response.json()
    assert body["code"] == "invalid_audio"
    assert body["details"][0]["field"] == "file"


async def test_parse_returns_422_when_confidence_is_below_the_threshold(
    settings, store, wav_bytes, monkeypatch
):
    monkeypatch.setattr("app.routes.transcribe", _stub_transcribe(TRANSCRIPT))
    monkeypatch.setattr(
        "app.routes.extract_assessment",
        _stub_extract(StubLLM(confidence=0.1, notes="values are not stated")),
    )
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_store] = lambda: store
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as http:
        response = await http.post(
            "/assessments/parse", files={"file": ("a.wav", wav_bytes, "audio/wav")}
        )
    app.dependency_overrides.clear()

    assert response.status_code == 422
    body = response.json()
    assert body["code"] == "low_extraction_confidence"
    assert body["details"][0]["field"] == "confidence"
    assert body["details"][0]["threshold"] == 0.6
    assert body["details"][0]["message"] == "values are not stated"


async def test_parse_reports_a_transcription_failure(settings, store, wav_bytes, monkeypatch):
    monkeypatch.setattr(
        "app.routes.transcribe",
        _stub_transcribe(
            "",
            PipelineError(
                "transcription_failed",
                "Local Whisper transcription failed.",
                502,
                [{"field": "file", "message": "could not decode the audio"}],
            ),
        ),
    )
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_store] = lambda: store
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as http:
        response = await http.post(
            "/assessments/parse", files={"file": ("a.wav", wav_bytes, "audio/wav")}
        )
    app.dependency_overrides.clear()

    assert response.status_code == 502
    assert response.json()["code"] == "transcription_failed"
    assert response.json()["details"][0]["message"] == "could not decode the audio"


async def test_save_get_and_list_roundtrip(client, store, wav_bytes):
    async with client as http:
        parsed = await http.post(
            "/assessments/parse", files={"file": ("a.wav", wav_bytes, "audio/wav")}
        )
        saved = await http.post(
            "/assessments",
            json={"assessment": parsed.json(), "metadata": {"sourceFile": "a.wav"}},
        )
        assessment_id = saved.json()["id"]
        fetched = await http.get(f"/assessments/{assessment_id}")
        listed = await http.get("/assessments")

    assert saved.status_code == 201
    assert fetched.status_code == 200
    assert fetched.json()["assessment"] == parsed.json()
    assert fetched.json()["metadata"] == {"sourceFile": "a.wav"}
    assert listed.json()["count"] == 1


async def test_list_filters_by_date(client, store):
    async with client as http:
        await http.post(
            "/assessments", json={"assessment": FirstAssessment().model_dump()}
        )
        store.records[0]["createdAt"] = datetime(2026, 1, 5, tzinfo=timezone.utc)

        january = await http.get(
            "/assessments", params={"from_date": "2026-01-01", "to_date": "2026-01-31"}
        )
        later = await http.get("/assessments", params={"from_date": "2026-06-01"})

    assert january.json()["count"] == 1
    assert later.json()["count"] == 0


async def test_save_rejects_a_payload_with_unknown_fields(client):
    async with client as http:
        response = await http.post(
            "/assessments", json={"assessment": {"clinicalDetails": {"onset": "8 months"}}}
        )

    assert response.status_code == 422
    body = response.json()
    assert body["code"] == "request_validation_failed"
    assert any("clinicalDetails" in detail["field"] for detail in body["details"])


async def test_get_an_unknown_id_returns_404(client):
    async with client as http:
        response = await http.get("/assessments/000000000000000000000042")

    assert response.status_code == 404
    assert response.json()["code"] == "assessment_not_found"


async def test_a_database_failure_returns_503(settings):
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_store] = lambda: FakeStore(
        fail=PipelineError(
            "database_unavailable",
            "Could not save the assessment.",
            503,
            [{"field": "database", "message": "connection refused"}],
        )
    )
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as http:
        response = await http.post(
            "/assessments", json={"assessment": FirstAssessment().model_dump()}
        )
    app.dependency_overrides.clear()

    assert response.status_code == 503
    assert response.json()["code"] == "database_unavailable"
