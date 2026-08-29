from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.assessments import router as assessments_router
from app.db.mongodb import close_mongo_connection, connect_to_mongo


@asynccontextmanager
async def lifespan(app: FastAPI):
    connect_to_mongo()
    yield
    close_mongo_connection()


app = FastAPI(
    title="Clinical Assessment Pipeline",
    description=(
        "WAV clinical recording -> Whisper transcription -> LangGraph "
        "clinical extraction -> validated FirstAssessment -> MongoDB."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(assessments_router)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
