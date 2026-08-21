from contextlib import asynccontextmanager

from fastapi import FastAPI
from motor.motor_asyncio import AsyncIOMotorClient

from app.api.assessments import get_repository, router
from app.config import get_settings
from app.db.mongo import get_repository as build_repository


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    client = AsyncIOMotorClient(settings.mongo_uri)
    repository = build_repository(client, settings.mongo_db_name)

    app.dependency_overrides[get_repository] = lambda: repository

    yield

    client.close()


app = FastAPI(title="Stance Health — Clinical Assessment Pipeline", lifespan=lifespan)
app.include_router(router)


@app.get("/health")
async def health():
    return {"status": "ok"}
