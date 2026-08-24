"""MongoDB connection management.

Uses Motor (async) against MongoDB. If Mongo is unreachable and ALLOW_MEMORY_DB is
set (default in dev/test), falls back to an in-process store implementing the same
async surface the repository uses — so the API and tests run without a live Mongo.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from app.config import settings


# --------------------------------------------------------------------------- #
# In-memory fallback collection (async-compatible subset of Motor's API)
# --------------------------------------------------------------------------- #
class _MemoryCollection:
    def __init__(self) -> None:
        self._docs: Dict[str, Dict[str, Any]] = {}

    async def insert_one(self, doc: Dict[str, Any]):
        _id = doc.get("_id") or str(uuid.uuid4())
        doc = {**doc, "_id": _id}
        self._docs[_id] = doc
        return type("Res", (), {"inserted_id": _id})()

    async def find_one(self, flt: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        _id = flt.get("_id")
        return self._docs.get(_id)

    def find(self, flt: Optional[Dict[str, Any]] = None):
        return _MemoryCursor(list(self._docs.values()), flt or {})


class _MemoryCursor:
    def __init__(self, docs: List[Dict[str, Any]], flt: Dict[str, Any]):
        self._docs = docs
        self._flt = flt

    def sort(self, key: str, direction: int = -1):
        reverse = direction < 0
        self._docs = sorted(self._docs, key=lambda d: d.get(key, ""), reverse=reverse)
        return self

    async def to_list(self, length: Optional[int] = None) -> List[Dict[str, Any]]:
        docs = self._docs
        created = self._flt.get("createdAt")
        if isinstance(created, dict):
            gte, lte = created.get("$gte"), created.get("$lte")
            if gte is not None:
                docs = [d for d in docs if d.get("createdAt") and d["createdAt"] >= gte]
            if lte is not None:
                docs = [d for d in docs if d.get("createdAt") and d["createdAt"] <= lte]
        return docs[:length] if length else docs


class Database:
    """Holds the active collection (real or in-memory) and its backend name."""

    def __init__(self) -> None:
        self.collection: Any = None
        self.backend: str = "uninitialised"
        self._client: Any = None

    async def connect(self) -> None:
        try:
            from motor.motor_asyncio import AsyncIOMotorClient

            client = AsyncIOMotorClient(settings.mongo_uri, serverSelectionTimeoutMS=1500)
            await client.admin.command("ping")
            self._client = client
            self.collection = client[settings.mongo_db][settings.mongo_collection]
            self.backend = "mongodb"
            return
        except Exception:
            if not settings.allow_memory_db:
                raise
            self.collection = _MemoryCollection()
            self.backend = "memory"

    async def close(self) -> None:
        if self._client is not None:
            self._client.close()

    @staticmethod
    def now() -> datetime:
        return datetime.now(timezone.utc)


db = Database()
