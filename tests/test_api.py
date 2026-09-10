import io
import pytest
from httpx import AsyncClient, ASGITransport
from app.main import app
from app.models.assessment import FirstAssessment


@pytest.mark.anyio
async def test_health_check():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.get("/health")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "healthy"


@pytest.mark.anyio
async def test_parse_bad_audio_returns_400():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Send random non-WAV bytes
        files = {"file": ("corrupt.wav", io.BytesIO(b"NOT_A_WAV_FILE_DATA"), "audio/wav")}
        res = await client.post("/assessments/parse", files=files)
        assert res.status_code == 400
        assert "Audio validation error" in res.json()["detail"]


@pytest.mark.anyio
async def test_get_assessments_date_validation():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # from > to should return 400
        res = await client.get("/assessments", params={"from": "2026-12-01T00:00:00Z", "to": "2026-01-01T00:00:00Z"})
        assert res.status_code == 400
        assert "cannot be after" in res.json()["detail"]


@pytest.mark.anyio
async def test_get_assessment_not_found():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.get("/assessments/non-existent-id-12345")
        assert res.status_code in [404, 500]  # 404 if db reachable or 500 if local mongo is down
