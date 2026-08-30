import pytest

from app.db.mongo import get_assessment_by_id
from app.errors import BadRequestError


@pytest.mark.asyncio
async def test_invalid_mongodb_id():
    with pytest.raises(BadRequestError):
        await get_assessment_by_id("not-an-object-id")


@pytest.mark.asyncio
async def test_missing_assessment(monkeypatch):
    class Collection:
        async def find_one(self, query):
            return None

    monkeypatch.setattr("app.db.mongo._collection", Collection())

    result = await get_assessment_by_id("64f0c0f4f4f4f4f4f4f4f4f4")

    assert result is None
