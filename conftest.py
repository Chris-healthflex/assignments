import sys
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from mongomock_motor import AsyncMongoMockClient

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from app.main import app

@pytest.fixture
def client():
    """Test client with mocked MongoDB."""
    mock_mongo = AsyncMongoMockClient()
    app.state.mongo_client = mock_mongo
    with TestClient(app) as test_client:
        yield test_client