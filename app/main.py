"""FastAPI application entrypoint.

Wires the router, opens/closes the Mongo (or in-memory) connection on lifespan,
and exposes a health endpoint that reports which DB backend is active.
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import router
from app.config import settings
from app.db.client import db


@asynccontextmanager
async def lifespan(app: FastAPI):
    await db.connect()
    yield
    await db.close()


app = FastAPI(
    title="Voice → Structured Clinical Assessment",
    description="Transcribes a clinician-patient WAV and extracts a FirstAssessment.",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(router)


@app.get("/health", summary="Liveness + active backends")
async def health():
    return {
        "status": "ok",
        "db_backend": db.backend,
        "llm_backend": "stub" if settings.use_stub_llm else settings.llm_backend,
        "whisper_backend": settings.whisper_backend,
    }
