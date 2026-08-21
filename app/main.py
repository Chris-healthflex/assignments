from contextlib import asynccontextmanager

import certifi
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo.errors import PyMongoError

from app.api.assessments import get_repository, router
from app.config import get_settings
from app.db.mongo import get_repository as build_repository


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    client_kwargs = {"serverSelectionTimeoutMS": 5000}
    if settings.mongo_uri.startswith("mongodb+srv://"):
        # Atlas requires TLS; some Python installs (notably python.org's
        # macOS build) ship without access to the system CA store, which
        # otherwise fails with CERTIFICATE_VERIFY_FAILED.
        client_kwargs["tlsCAFile"] = certifi.where()

    client = AsyncIOMotorClient(settings.mongo_uri, **client_kwargs)
    repository = build_repository(client, settings.mongo_db_name)

    app.dependency_overrides[get_repository] = lambda: repository

    yield

    client.close()


def create_app(lifespan=lifespan) -> FastAPI:
    app = FastAPI(title="Stance Health — Clinical Assessment Pipeline", lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router)

    @app.exception_handler(PyMongoError)
    async def mongo_error_handler(request: Request, exc: PyMongoError):
        return JSONResponse(status_code=503, content={"detail": "Database unavailable"})

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    return app


app = create_app()
