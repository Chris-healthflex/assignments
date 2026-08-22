"""MongoDB connection lifecycle (D4).

One client is created when the app starts and closed when it stops. Motor
pools connections internally, so creating a client per request would waste the
pool and leak sockets.

``set_client`` exists so tests can inject ``mongomock_motor`` and run the whole
repository and API suite without a live server.
"""

from __future__ import annotations

import logging

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)

_client = None
_settings: Settings | None = None


class DatabaseUnavailableError(RuntimeError):
    """MongoDB could not be reached.

    Surfaces as HTTP 503 with the URI that was tried, so the caller can tell a
    stopped server apart from a wrong connection string.
    """


def set_client(client, settings: Settings | None = None) -> None:
    """Install a client directly. Used by tests to inject an in-memory Mongo."""
    global _client, _settings
    _client = client
    _settings = settings or get_settings()


async def connect(settings: Settings | None = None) -> None:
    """Open the connection pool and ensure indexes exist."""
    global _client, _settings
    _settings = settings or get_settings()

    if _client is not None:
        return

    try:
        from motor.motor_asyncio import AsyncIOMotorClient
    except ImportError as exc:  # pragma: no cover - required dependency
        raise DatabaseUnavailableError(
            "motor is not installed. Run: pip install -r requirements.txt"
        ) from exc

    _client = AsyncIOMotorClient(
        _settings.mongodb_uri,
        serverSelectionTimeoutMS=_settings.mongodb_timeout_ms,
        uuidRepresentation="standard",
    )
    logger.info("Connected to MongoDB at %s", _settings.mongodb_uri)
    await ensure_indexes()


async def disconnect() -> None:
    global _client
    if _client is not None:
        close = getattr(_client, "close", None)
        if close is not None:
            result = close()
            if hasattr(result, "__await__"):
                await result
        _client = None
        logger.info("Disconnected from MongoDB")


def get_collection():
    """The collection assessments are stored in."""
    if _client is None:
        raise DatabaseUnavailableError(
            "MongoDB client is not initialised. The application lifespan should "
            "call connect() on startup."
        )
    settings = _settings or get_settings()
    return _client[settings.mongodb_db][settings.mongodb_collection]


async def ensure_indexes() -> None:
    """Create the indexes the API relies on.

    ``createdAt`` is descending because every listing is newest-first and the
    date filter in EP4 ranges over it; without it that endpoint degrades to a
    collection scan as history grows.
    """
    try:
        await get_collection().create_index([("createdAt", -1)], name="createdAt_desc")
    except Exception as exc:
        # An index failure must not stop the service from starting; queries
        # still work, just more slowly.
        logger.warning("Could not create indexes: %s", exc)


async def ping() -> bool:
    """Health-check probe. Returns False rather than raising."""
    if _client is None:
        return False
    try:
        await _client.admin.command("ping")
        return True
    except Exception as exc:
        logger.warning("MongoDB ping failed: %s", exc)
        return False
