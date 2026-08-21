from app.db.client import get_database
from app.core.config import settings

def get_db():
    return get_database()