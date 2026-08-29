"""FastAPI application entrypoint."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import router
from app.db.connection import connect, disconnect

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        await connect()
    except Exception:
        logger.exception("MongoDB unavailable at startup; DB endpoints will fail")
    yield
    await disconnect()


app = FastAPI(
    title="Clinical Assessment Pipeline",
    description="WAV -> Whisper -> LangGraph -> FirstAssessment -> MongoDB",
    version="1.0.0",
    lifespan=lifespan,
)
app.include_router(router)


@app.get("/health", tags=["ops"])
async def health() -> dict:
    return {"status": "ok"}
