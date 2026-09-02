"""Async MongoDB connection manager using Motor."""

from typing import Optional
from motor.motor_asyncio import (
    AsyncIOMotorClient,
    AsyncIOMotorCollection,
    AsyncIOMotorDatabase,
)
from pymongo.errors import ConnectionFailure, PyMongoError

from app.config import settings


class DatabaseException(Exception):
    """Base exception for database-related errors."""

    pass


class DatabaseConnectionError(DatabaseException):
    """Raised when connecting to MongoDB fails."""

    pass


class MongoDBManager:
    """Singleton connection manager for Async MongoDB access."""

    def __init__(
        self,
        uri: Optional[str] = None,
        db_name: Optional[str] = None,
        collection_name: Optional[str] = None,
    ) -> None:
        """Initialize MongoDBManager with application settings.

        Args:
            uri: Optional MongoDB connection URI override.
            db_name: Optional database name override.
            collection_name: Optional collection name override.
        """
        self.uri = uri or settings.MONGO_URI
        self.db_name = db_name or settings.MONGO_DB_NAME
        self.collection_name = collection_name or settings.MONGO_COLLECTION
        self._client: Optional[AsyncIOMotorClient] = None

    @property
    def client(self) -> AsyncIOMotorClient:
        """Get or lazily initialize the AsyncIOMotorClient."""
        if self._client is None:
            self._client = AsyncIOMotorClient(
                self.uri,
                serverSelectionTimeoutMS=3000,
                connectTimeoutMS=3000,
            )
        return self._client

    def get_database(self) -> AsyncIOMotorDatabase:
        """Get the configured AsyncIOMotorDatabase instance."""
        return self.client[self.db_name]

    def get_collection(self, name: Optional[str] = None) -> AsyncIOMotorCollection:
        """Get the configured AsyncIOMotorCollection instance."""
        col_name = name or self.collection_name
        return self.get_database()[col_name]

    async def ping(self) -> bool:
        """Check MongoDB connectivity by pinging the admin database.

        Returns:
            True if connection is healthy.

        Raises:
            DatabaseConnectionError: If ping fails.
        """
        try:
            pong = await self.client.admin.command("ping")
            return bool(pong.get("ok", 0) == 1.0)
        except (ConnectionFailure, PyMongoError, Exception) as exc:
            raise DatabaseConnectionError(f"Failed to connect to MongoDB at {self.uri}: {str(exc)}") from exc

    def close(self) -> None:
        """Close the MongoDB client connection cleanly."""
        if self._client is not None:
            self._client.close()
            self._client = None


# Global singleton instance
db_manager = MongoDBManager()
