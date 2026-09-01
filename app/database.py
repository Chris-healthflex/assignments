import os

from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv(override=True)

MONGODB_URI = os.getenv("MONGODB_URI")
MONGODB_DATABASE = os.getenv(
    "MONGODB_DATABASE",
    "clinical_assessments"
)

if not MONGODB_URI:
    raise ValueError("MONGODB_URI is not set in .env")

client = MongoClient(MONGODB_URI)

db = client[MONGODB_DATABASE]

assessments_collection = db["assessments"]