"""FastAPI application entry point (D1).

Run with:
    uvicorn app.main:app --reload

Two front doors, both first-class:

* ``/``      the clinician interface - upload a recording, review the extracted
             record, export it as a PDF, sign it off.
* ``/docs``  Swagger UI over the same API. The OpenAPI metadata is written to
             be read, not left as boilerplate, since a reviewer may arrive here
             rather than at the interface.

The interface is static files with no build step, so it can be read without a
toolchain.
"""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.config import PROJECT_ROOT, get_settings
from app.db import client as db_client

STATIC_DIR = PROJECT_ROOT / "app" / "static"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

DESCRIPTION = """
Turns a WAV recording of a clinician-patient session into a structured
assessment in the exact **FirstAssessment** schema, and stores it in MongoDB.

### Pipeline

`WAV upload` -> `Whisper transcription` -> `LangGraph extraction` ->
`grounding verification` -> `FirstAssessment JSON` -> `MongoDB`

### How hallucination is prevented

Extraction is followed by a deterministic grounding pass that uses no LLM. A
value survives only if every number it contains appears in the transcript,
every date-like token appears in the transcript, and enough of its content
words appear that it reads as a transcription rather than an invention.

Anything that fails is **cleared to an empty string and reported in
`flaggedFields`** - never kept, and never replaced with a guess. A blank
flagged field is safe for a clinician to complete; a confident wrong
measurement is not.

`flaggedFields` separates two cases:

* `not_stated` - the recording never covered this field. Benign and expected.
* `rejected` - the model produced a value that failed verification. This is a
  caught hallucination, and the discarded value is included for audit.

### Interfaces

The clinician interface is at `/`. This page documents the API behind it.

### Timing

Parsing is slow on local models: roughly 25 s of transcription plus around
two minutes of extraction for a two-minute recording. The Whisper model loads
on first use, so the very first request is slower still.
"""

TAGS = [
    {"name": "assessments", "description": "Parse, store and retrieve clinical assessments."},
    {"name": "system", "description": "Health and dependency checks."},
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    try:
        await db_client.connect(settings)
    except Exception as exc:
        # Start anyway. Parsing does not need MongoDB, and /docs plus /health
        # are far more useful than a process that refuses to boot.
        logger.warning("MongoDB unavailable at startup: %s", exc)

    logger.info(
        "Ready. LLM %s/%s, Whisper %s/%s",
        settings.llm_provider, settings.default_llm_model,
        settings.whisper_backend, settings.whisper_model,
    )
    yield
    await db_client.disconnect()


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        description=DESCRIPTION,
        version=settings.api_version,
        openapi_tags=TAGS,
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def add_request_id(request: Request, call_next):
        """Tag each request so slow pipeline stages can be traced in the log."""
        request_id = str(uuid.uuid4())[:8]
        started = time.perf_counter()
        response = await call_next(request)
        elapsed = time.perf_counter() - started
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time"] = f"{elapsed:.3f}"
        logger.info(
            "[%s] %s %s -> %s in %.2fs",
            request_id, request.method, request.url.path, response.status_code, elapsed,
        )
        return response

    @app.exception_handler(Exception)
    async def unhandled_exception(request: Request, exc: Exception):
        """Never leak a stack trace to a caller; log it instead."""
        logger.exception("Unhandled error on %s %s", request.method, request.url.path)
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error. See server logs for details."},
        )

    app.include_router(router)

    # The clinician interface. Static files only - no build step, no bundler -
    # so the front end can be read without a toolchain. Mounted after the
    # router so an API path can never be shadowed by a file of the same name.
    if STATIC_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

        @app.get("/", include_in_schema=False)
        async def index():
            return FileResponse(STATIC_DIR / "index.html")

        @app.get("/favicon.ico", include_in_schema=False)
        async def favicon():
            return Response(status_code=204)
    return app


app = create_app()
