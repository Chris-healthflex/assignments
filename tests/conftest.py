"""Shared fixtures: fake transcriber, in-memory assessment store, test client."""
from __future__ import annotations

import os
import sys
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest
from bson import ObjectId
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("CONFIDENCE_THRESHOLD", "0.70")

from app.api import routes as routes_module  # noqa: E402
from app.main import app  # noqa: E402
from app.models.assessment import SECTION_ALIASES, FirstAssessment  # noqa: E402
from app.models.internal import ExtractionEnvelope  # noqa: E402
from app.transcription.whisper_service import TranscriptionError  # noqa: E402

SAMPLE_TRANSCRIPT = (
    "Clinician: What brings you in today? "
    "Patient: I've had sharp pain in my right knee for about three weeks."
)


def make_assessment() -> FirstAssessment:
    """A representative schema/v1 assessment, built from camelCase input."""
    return FirstAssessment.model_validate(
        {
            "clinicalDetails": {
                "clinicalHistory": "No prior knee injury or surgery.",
                "chiefComplaint": "Sharp pain in the right knee",
                "duration": "3 weeks",
            },
            "subjectiveAssessments": [
                {"testName": "Pain scale", "conclusion": "7/10 on stairs"}
            ],
            "objectiveAssessment": {
                "tests": [
                    {
                        "testName": "Knee flexion ROM",
                        "unitName": "degrees",
                        "value": "",
                        "left": "135",
                        "right": "110",
                        "comments": "Restricted on the right",
                    }
                ]
            },
            "subjectiveGoals": [
                {"goalDetails": "Climb stairs without pain", "targetDate": "2026-10-01"}
            ],
            "objectiveGoals": [
                {
                    "goalName": "Right knee flexion",
                    "goalCategory": "Range of motion",
                    "unitName": "degrees",
                    "value": "135",
                    "targetDate": "2026-10-01",
                }
            ],
            "recommendation": [
                {"sessionType": "Physiotherapy", "sessionFrequency": "Twice weekly"}
            ],
            "patientAdvice": {"adviceDetails": "Ice for 15 minutes after activity."},
        }
    )


def full_confidence(score: float = 0.95) -> Dict[str, float]:
    """Confidence map keyed by the seven camelCase section names."""
    return {alias: score for alias in SECTION_ALIASES.values()}


class FakeLLM:
    """Stands in for ChatOpenAI(...).with_structured_output(ExtractionEnvelope)."""

    def __init__(self, envelope: Optional[ExtractionEnvelope] = None, raises: bool = False):
        self.envelope = envelope or ExtractionEnvelope(
            assessment=make_assessment(), field_confidence=full_confidence()
        )
        self.raises = raises

    def invoke(self, messages):
        if self.raises:
            raise RuntimeError("LLM provider unavailable")
        return self.envelope


class FakeTranscriber:
    def __init__(self, text: str = SAMPLE_TRANSCRIPT, error: Optional[str] = None):
        self.text = text
        self.error = error

    def transcribe(self, audio_path):
        if self.error:
            raise TranscriptionError(self.error)
        return self.text


class FakeCursor:
    def __init__(self, docs: List[Dict[str, Any]]):
        self._docs = docs

    def sort(self, key, direction):
        self._docs = sorted(
            self._docs, key=lambda d: d["created_at"], reverse=direction < 0
        )
        return self

    def skip(self, n):
        self._docs = self._docs[n:]
        return self

    def limit(self, n):
        self._docs = self._docs[:n]
        return self

    def __aiter__(self):
        async def gen():
            for doc in self._docs:
                yield doc

        return gen()


class FakeCollection:
    """Minimal async Mongo collection - avoids needing a live server for unit tests."""

    def __init__(self):
        self.docs: List[Dict[str, Any]] = []

    async def insert_one(self, document):
        document = dict(document)
        document["_id"] = ObjectId()
        self.docs.append(document)

        class Result:
            inserted_id = document["_id"]

        return Result()

    async def find_one(self, query):
        for doc in self.docs:
            if doc["_id"] == query["_id"]:
                return doc
        return None

    def find(self, query):
        window = query.get("created_at")
        selected = self.docs
        if window:
            def keep(doc):
                created = doc["created_at"]
                if "$gte" in window and created < window["$gte"]:
                    return False
                if "$lt" in window and created >= window["$lt"]:
                    return False
                return True

            selected = [d for d in selected if keep(d)]
        return FakeCursor(list(selected))


@pytest.fixture
def fake_collection():
    return FakeCollection()


@pytest.fixture
def fake_transcriber():
    return FakeTranscriber()


@pytest.fixture
def client(fake_collection, fake_transcriber, monkeypatch):
    from app.db.repository import AssessmentRepository

    routes_module.app_test_collection = fake_collection
    app.dependency_overrides[routes_module.get_repository] = lambda: AssessmentRepository(
        collection=fake_collection
    )
    app.dependency_overrides[routes_module.get_transcriber] = lambda: fake_transcriber

    # Route the graph's LLM through a fake so no network call is made.
    monkeypatch.setattr(routes_module, "run_extraction", _patched_run_extraction)

    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


_llm_for_tests = FakeLLM()


def _patched_run_extraction(transcript, llm=None, threshold=None):
    from app.agent.graph import run_extraction as real

    return real(transcript, llm=_llm_for_tests, threshold=threshold)


@pytest.fixture
def set_llm():
    def _set(envelope=None, raises=False):
        global _llm_for_tests
        _llm_for_tests.envelope = envelope or _llm_for_tests.envelope
        _llm_for_tests.raises = raises
        return _llm_for_tests

    yield _set
    _llm_for_tests.envelope = ExtractionEnvelope(
        assessment=make_assessment(), field_confidence=full_confidence()
    )
    _llm_for_tests.raises = False


@pytest.fixture
def wav_bytes(tmp_path):
    """A tiny but structurally valid WAV file."""
    path = tmp_path / "sample.wav"
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\x00\x00" * 16000)
    return path.read_bytes()
