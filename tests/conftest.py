import os
import pytest
import sys
from unittest.mock import MagicMock, patch

# Force test configuration in environment before any import loads them
os.environ["MONGODB_URI"] = "mongodb://mock-connection:27017"
os.environ["MONGODB_DATABASE"] = "test_clinical_assessment"
os.environ["GROQ_API_KEY"] = "mock-groq-key"
os.environ["CONFIDENCE_THRESHOLD"] = "0.70"

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock MongoClient so it doesn't try to connect during startup or test runs
@pytest.fixture(autouse=True)
def mock_mongo_client():
    with patch("app.database.mongodb.MongoClient") as mock_client:
        mock_instance = MagicMock()
        mock_client.return_value = mock_instance
        # Mock ping command
        mock_instance.admin.command.return_value = {"ok": 1.0}
        yield mock_instance

# Mock LLM calls
@pytest.fixture
def mock_llm():
    with patch("app.services.extraction.get_llm") as mock_get_llm:
        mock_llm_instance = MagicMock()
        mock_get_llm.return_value = mock_llm_instance
        yield mock_llm_instance

# Test Client fixture that enters lifespan
@pytest.fixture
def client():
    from app.main import app
    from fastapi.testclient import TestClient
    with TestClient(app) as c:
        yield c

