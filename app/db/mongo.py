from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from bson import ObjectId
from pymongo import MongoClient
from pymongo.collection import Collection

from app.config import get_settings


_client: MongoClient | None = None


COLLECTION_NAME = "assessments"


def get_client() -> MongoClient:
    global _client

    if _client is None:
        settings = get_settings()

        _client = MongoClient(
            settings.mongodb_uri,
            serverSelectionTimeoutMS=5000,
        )

    return _client


def get_collection() -> Collection:
    settings = get_settings()

    client = get_client()

    database = client[
        settings.mongodb_database
    ]

    return database[
        COLLECTION_NAME
    ]


def check_mongodb() -> bool:
    try:
        get_client().admin.command("ping")
        return True
    except Exception:
        return False


def save_assessment(
    assessment: dict[str, Any],
) -> str:

    collection = get_collection()

    document = {
        "assessment": assessment,
        "createdAt": datetime.now(
            timezone.utc
        ),
    }

    result = collection.insert_one(
        document
    )

    return str(result.inserted_id)


def get_assessment(
    assessment_id: str,
) -> dict[str, Any] | None:

    collection = get_collection()

    try:
        object_id = ObjectId(
            assessment_id
        )
    except Exception:
        return None

    document = collection.find_one(
        {"_id": object_id}
    )

    if document is None:
        return None

    return {
        "id": str(
            document["_id"]
        ),
        "assessment": document[
            "assessment"
        ],
        "createdAt": document[
            "createdAt"
        ].isoformat(),
    }


def list_assessments(
    date_from: datetime | None = None,
    date_to: datetime | None = None,
) -> list[dict[str, Any]]:

    collection = get_collection()

    query: dict[str, Any] = {}

    if date_from or date_to:

        created_query: dict[str, Any] = {}

        if date_from:
            created_query["$gte"] = date_from

        if date_to:
            created_query["$lte"] = date_to

        query["createdAt"] = created_query

    cursor = collection.find(
        query
    ).sort(
        "createdAt",
        -1,
    )

    results = []

    for document in cursor:

        results.append(
            {
                "id": str(
                    document["_id"]
                ),
                "assessment": document[
                    "assessment"
                ],
                "createdAt": document[
                    "createdAt"
                ].isoformat(),
            }
        )

    return results
