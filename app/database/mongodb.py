from typing import Any
from pymongo import MongoClient
from pymongo.database import Database
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
from app.core.config import get_settings
from app.core.logging import logger
from app.core.errors import DatabaseError


class MongoDBManager:
    """Manages MongoDB client and database connections."""

    def __init__(self) -> None:
        self._client: MongoClient[dict[str, Any]] | None = None
        self._db: Database[dict[str, Any]] | None = None

    def connect(self) -> None:
        """Initializes client connection to MongoDB."""
        settings = get_settings()
        if self._client is not None:
            return

        try:
            logger.info("Connecting to MongoDB at %s", settings.MONGODB_URI)
            self._client = MongoClient(
                settings.MONGODB_URI,
                serverSelectionTimeoutMS=settings.MONGODB_TIMEOUT_MS,
                connectTimeoutMS=settings.MONGODB_TIMEOUT_MS,
                maxPoolSize=50,
                minPoolSize=5,
            )
            # Trigger quick server check
            self._client.admin.command("ping")
            self._db = self._client[settings.MONGODB_DATABASE]
            logger.info("Successfully connected to MongoDB database '%s'", settings.MONGODB_DATABASE)
        except (ConnectionFailure, ServerSelectionTimeoutError) as exc:
            logger.warning("MongoDB initial connection check failed: %s (will retry on query or use mock)", exc)
            self._db = self._client[settings.MONGODB_DATABASE] if self._client else None

    def close(self) -> None:
        """Closes the MongoDB connection."""
        if self._client:
            logger.info("Closing MongoDB connection")
            self._client.close()
            self._client = None
            self._db = None

    def get_database(self) -> Database[dict[str, Any]]:
        """Returns active database instance."""
        if self._db is None or self._client is None:
            self.connect()
        if self._db is None:
            raise DatabaseError("MongoDB is unavailable. Database handle is not initialized.")
        return self._db

    def set_client_override(self, client: Any) -> None:
        """Allows injecting a mock client (e.g. mongomock) during testing."""
        settings = get_settings()
        self._client = client
        self._db = client[settings.MONGODB_DATABASE]


db_manager = MongoDBManager()


def get_db() -> Database[dict[str, Any]]:
    """Dependency provider for database instance."""
    return db_manager.get_database()
