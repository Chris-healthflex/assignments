"""Endpoint tests. No live MongoDB, Whisper model, or LLM required."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from tests.conftest import FakeTranscriber, full_confidence, make_assessment
from app.models.internal import ExtractionEnvelope


# ---------- POST /assessments/parse ----------

def test_parse_returns_first_assessment(client, wav_bytes):
    response = client.post(
        "/assessments/parse",
        files={"file": ("session.wav", wav_bytes, "audio/wav")},
    )
    assert response.status_code == 200
    body = response.json()

    # Exact schema/v1 shape: seven camelCase sections, nothing more.
    assert set(body) == {
        "clinicalDetails",
        "subjectiveAssessments",
        "objectiveAssessment",
        "subjectiveGoals",
        "objectiveGoals",
        "recommendation",
        "patientAdvice",
    }
    assert body["clinicalDetails"]["chiefComplaint"] == "Sharp pain in the right knee"
    assert body["objectiveAssessment"]["tests"][0]["right"] == "110"
    # Arrays stay arrays; strings are never null.
    assert isinstance(body["recommendation"], list)
    assert body["patientAdvice"]["adviceDetails"] is not None
    # Confidence metadata must never leak into the frontend payload.
    assert "field_confidence" not in body
    assert "confidence" not in body


def test_parse_rejects_non_wav(client):
    response = client.post(
        "/assessments/parse",
        files={"file": ("notes.mp3", b"not audio", "audio/mpeg")},
    )
    assert response.status_code == 400
    assert ".wav" in response.json()["detail"]


def test_parse_handles_transcription_failure(client, wav_bytes, fake_transcriber):
    fake_transcriber.error = "Not a valid WAV file: corrupt header"
    response = client.post(
        "/assessments/parse",
        files={"file": ("broken.wav", wav_bytes, "audio/wav")},
    )
    assert response.status_code == 400
    assert "corrupt header" in response.json()["detail"]


def test_parse_low_confidence_returns_422_with_field_detail(
    client, wav_bytes, set_llm
):
    scores = full_confidence()
    scores["clinicalDetails"] = 0.42
    scores["objectiveGoals"] = 0.31
    set_llm(ExtractionEnvelope(assessment=make_assessment(), field_confidence=scores))

    response = client.post(
        "/assessments/parse",
        files={"file": ("session.wav", wav_bytes, "audio/wav")},
    )
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail["threshold"] == 0.70
    failing = {f["field"]: f["confidence"] for f in detail["fields"]}
    assert failing == {"clinicalDetails": 0.42, "objectiveGoals": 0.31}


def test_parse_handles_llm_failure(client, wav_bytes, set_llm):
    set_llm(raises=True)
    response = client.post(
        "/assessments/parse",
        files={"file": ("session.wav", wav_bytes, "audio/wav")},
    )
    assert response.status_code == 502


# ---------- POST /assessments ----------

def test_create_assessment_persists(client):
    payload = {"assessment": make_assessment().model_dump(mode="json")}
    response = client.post("/assessments", json=payload)
    assert response.status_code == 201
    body = response.json()
    assert body["id"]
    assert body["assessment"]["clinicalDetails"]["chiefComplaint"] == (
        "Sharp pain in the right knee"
    )
    assert body["created_at"]


def test_create_assessment_rejects_invalid_body(client):
    response = client.post(
        "/assessments", json={"assessment": {"objectiveAssessment": "not-an-object"}}
    )
    assert response.status_code == 422


# ---------- GET /assessments/{id} ----------

def test_get_by_id_roundtrip(client):
    created = client.post(
        "/assessments", json={"assessment": make_assessment().model_dump(mode="json")}
    ).json()
    response = client.get(f"/assessments/{created['id']}")
    assert response.status_code == 200
    assert response.json()["id"] == created["id"]


def test_get_by_id_not_found(client):
    response = client.get("/assessments/507f1f77bcf86cd799439011")
    assert response.status_code == 404


def test_get_by_id_malformed(client):
    response = client.get("/assessments/not-an-object-id")
    assert response.status_code == 400


# ---------- GET /assessments ----------

def test_list_assessments(client):
    for _ in range(3):
        client.post(
            "/assessments",
            json={"assessment": make_assessment().model_dump(mode="json")},
        )
    response = client.get("/assessments")
    assert response.status_code == 200
    assert len(response.json()) == 3


def test_list_filters_by_date(client, fake_collection):
    client.post(
        "/assessments", json={"assessment": make_assessment().model_dump(mode="json")}
    )
    # Backdate the stored document by two days.
    fake_collection.docs[0]["created_at"] = datetime.now(timezone.utc) - timedelta(days=2)

    today = datetime.now(timezone.utc).date().isoformat()
    assert client.get(f"/assessments?date={today}").json() == []

    two_days_ago = (datetime.now(timezone.utc) - timedelta(days=2)).date().isoformat()
    assert len(client.get(f"/assessments?date={two_days_ago}").json()) == 1


def test_list_rejects_bad_date_format(client):
    response = client.get("/assessments?date=25-08-2026")
    assert response.status_code == 422
