from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_parse_rejects_wrong_file_type():
    response = client.post(
        "/assessments/parse",
        files={"file": ("notes.txt", b"not audio", "text/plain")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "File must be a WAV recording"


def test_parse_rejects_non_wav_content():
    response = client.post(
        "/assessments/parse",
        files={"file": ("audio.wav", b"not a wav file", "audio/wav")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "File must be a valid WAV recording"


def test_get_assessment_not_found(monkeypatch):
    async def fake_get_assessment_by_id(assessment_id: str):
        return None

    monkeypatch.setattr("app.routes.assessments.get_assessment_by_id", fake_get_assessment_by_id)

    response = client.get("/assessments/64f0c0f4f4f4f4f4f4f4f4f4")

    assert response.status_code == 404
    assert response.json()["detail"] == "Assessment not found"
