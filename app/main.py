from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.db.mongo import close_mongo, connect_mongo
from app.errors import DatabaseError, DatabaseUnavailableError
from app.routes.assessments import router as assessments_router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        await connect_mongo(get_settings())
    except DatabaseUnavailableError:
        logger.warning("MongoDB is unavailable at startup")

    yield
    await close_mongo()


app = FastAPI(title="Stance Assessment Pipeline", lifespan=lifespan)
app.include_router(assessments_router)


@app.exception_handler(DatabaseUnavailableError)
async def database_unavailable_handler(request: Request, exc: DatabaseUnavailableError):
    return JSONResponse(status_code=503, content={"detail": "Database is unavailable"})


@app.exception_handler(DatabaseError)
async def database_error_handler(request: Request, exc: DatabaseError):
    logger.exception("Database operation failed")
    return JSONResponse(status_code=500, content={"detail": "Database operation failed"})


@app.exception_handler(Exception)
async def unexpected_error_handler(request: Request, exc: Exception):
    if isinstance(exc, HTTPException):
        raise exc

    logger.exception("Unexpected server error")
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})
