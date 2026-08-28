"""
Endpoint-level tests for POST /assessments, GET /assessments/{id}, GET /assessments.

Uses an in-memory fake collection instead of a real MongoDB instance (none is
available in CI/sandbox here) — this validates the save/retrieve/list *logic* in
app/db/models.py and the endpoint wiring in app/routers/assessments.py, not the
Mongo driver itself. Point MONGO_URI at a real instance to test against real Mongo.
"""
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from fastapi.testclient import TestClient


class FakeCursor:
    def __init__(self, docs):
        self._docs = docs

    def sort(self, *a, **k):
        return self

    def skip(self, n):
        self._docs = self._docs[n:]
        return self

    def limit(self, n):
        self._docs = self._docs[:n]
        return self

    def __aiter__(self):
        self._iter = iter(self._docs)
        return self

    async def __anext__(self):
        try:
            return next(self._iter)
        except StopIteration:
            raise StopAsyncIteration


class FakeCollection:
    def __init__(self):
        self.store = {}

    async def insert_one(self, doc):
        self.store[doc["_id"]] = doc
        return doc

    async def find_one(self, query):
        return self.store.get(query.get("_id"))

    def find(self, query=None):
        docs = list(self.store.values())
        return FakeCursor(docs)


@pytest.fixture
def client(monkeypatch):
    fake_coll = FakeCollection()
    import app.db.models as models_mod

    monkeypatch.setattr(models_mod, "get_assessments_collection", lambda: fake_coll)

    from app.main import app

    return TestClient(app)


def _sample_assessment_payload():
    with open(
        os.path.join(os.path.dirname(__file__), "..", "sample_output", "clinical_assessment_output.json")
    ) as f:
        data = json.load(f)
    return data


def test_create_and_get_assessment(client):
    payload = _sample_assessment_payload()
    r = client.post("/assessments", json=payload)
    assert r.status_code == 200, r.text
    assessment_id = r.json()["id"]

    r2 = client.get(f"/assessments/{assessment_id}")
    assert r2.status_code == 200, r2.text
    body = r2.json()
    assert body["assessment"]["clinicalDetails"]["chiefComplaint"] == payload["assessment"]["clinicalDetails"]["chiefComplaint"]
    assert isinstance(body["assessment"]["subjectiveAssessments"], list)


def test_get_missing_assessment_404(client):
    r = client.get("/assessments/does-not-exist")
    assert r.status_code == 404


def test_list_assessments(client):
    payload = _sample_assessment_payload()
    client.post("/assessments", json=payload)
    client.post("/assessments", json=payload)

    r = client.get("/assessments")
    assert r.status_code == 200
    assert len(r.json()) == 2


def test_create_assessment_rejects_bad_schema(client):
    r = client.post("/assessments", json={"assessment": {"unexpectedField": True}})
    assert r.status_code == 422
