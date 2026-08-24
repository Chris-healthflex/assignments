"""API endpoint tests via FastAPI TestClient."""
import io, wave, struct, math
import pytest
from fastapi.testclient import TestClient
from app.main import app


def _wav_bytes():
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
        for i in range(16000):
            w.writeframes(struct.pack("<h", int(1000*math.sin(2*math.pi*220*i/16000))))
    return buf.getvalue()


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_transcribe_assess_rejects_non_wav(client):
    r = client.post("/transcribe-assess",
                    files={"file": ("x.txt", b"hello", "text/plain")})
    assert r.status_code == 422


def test_save_get_list_flow(client):
    payload = {"assessment": {"clinicalDetails": {"duration": "eight months"}}}
    r = client.post("/assessments", json=payload)
    assert r.status_code == 201
    new_id = r.json()["id"]

    r = client.get(f"/assessments/{new_id}")
    assert r.status_code == 200
    assert r.json()["assessment"]["clinicalDetails"]["duration"] == "eight months"

    r = client.get("/assessments")
    assert r.status_code == 200
    assert r.json()["count"] >= 1


def test_get_missing_returns_404(client):
    r = client.get("/assessments/nonexistent-id")
    assert r.status_code == 404


def test_list_bad_date_422(client):
    r = client.get("/assessments", params={"start_date": "not-a-date"})
    assert r.status_code == 422
