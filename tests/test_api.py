from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_health():
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "ok"


def test_parse_rejects_non_wav():
    response = client.post(
        "/assessments/parse",
        files={
            "file": (
                "audio.mp3",
                b"not really an audio file",
                "audio/mpeg",
            )
        },
    )

    assert response.status_code == 422


def test_parse_rejects_wrong_extension():
    response = client.post(
        "/assessments/parse",
        files={
            "file": (
                "audio.txt",
                b"hello",
                "text/plain",
            )
        },
    )

    assert response.status_code == 422
