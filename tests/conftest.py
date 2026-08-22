"""Shared test fixtures.

The database fixture swaps in ``mongomock_motor`` so the repository and API
suites run with no MongoDB server, no network, and no shared state between
tests. The production code path is unchanged - only the client object differs.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.config import get_settings
from app.db import client as db_client


@pytest.fixture
def mongo():
    """An empty in-memory MongoDB for the duration of one test."""
    from mongomock_motor import AsyncMongoMockClient

    client = AsyncMongoMockClient()
    db_client.set_client(client, get_settings())
    yield client
    db_client.set_client(None, get_settings())


@pytest.fixture
def transcript() -> str:
    """The cached real transcript when present, else a representative stand-in.

    Tests must not depend on Whisper having been run, but they benefit from
    the real text when it is available.
    """
    cached = Path(__file__).resolve().parent.parent / "data" / "transcript.txt"
    if cached.exists():
        return cached.read_text(encoding="utf-8")
    return (
        "The patient presented with left knee pain following surgery eight "
        "months ago after a road traffic accident resulting in a left tibial "
        "condyle fracture. Left knee flexion was 124 degrees compared with 130 "
        "degrees on the right. Physiotherapy was recommended once weekly for "
        "four sessions."
    )


@pytest.fixture
def sample_assessment():
    from app.schemas.first_assessment import FirstAssessment

    return FirstAssessment.model_validate(
        {
            "clinicalDetails": {
                "clinicalHistory": "road traffic accident",
                "chiefComplaint": "left knee pain",
                "duration": "eight months",
            },
            "subjectiveAssessments": [{"testName": "Pain", "conclusion": "moderate"}],
            "objectiveAssessment": {
                "tests": [
                    {
                        "testName": "Knee flexion",
                        "unitName": "degrees",
                        "left": "124",
                        "right": "130",
                    }
                ]
            },
            "recommendation": [
                {"sessionType": "Physiotherapy", "sessionFrequency": "once weekly"}
            ],
        }
    )
