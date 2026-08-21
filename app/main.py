from fastapi import FastAPI
from contextlib import asynccontextmanager

from app.core.config import settings
from app.api.routes import parse, assessments
from app.db.client import get_mongo_client

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    client = get_mongo_client()
    yield
    # Shutdown
    client.close()

app = FastAPI(
    title=settings.APP_NAME,
    version="1.0.0",
    lifespan=lifespan
)

app.include_router(parse.router, prefix=settings.API_V1_PREFIX)
app.include_router(assessments.router, prefix=settings.API_V1_PREFIX)

@app.get("/health")
async def health():
    return {"status": "ok"}