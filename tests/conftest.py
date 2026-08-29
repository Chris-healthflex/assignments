import os

os.environ.setdefault("WHISPER_MODEL_SIZE", "tiny")
os.environ.setdefault("GEMINI_API_KEY", "dummy-test-key")
os.environ.setdefault("MONGODB_URI", "mongodb://localhost:27017")
os.environ.setdefault("MONGODB_DB_NAME", "clinical_assessments_test")
