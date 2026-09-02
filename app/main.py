"""Main FastAPI application entrypoint."""

from contextlib import asynccontextmanager
from typing import AsyncGenerator, Dict
from fastapi import FastAPI, status

from app.api.assessments import router as assessments_router
from app.config import settings
from app.db.mongo import db_manager
from app.repositories.assessment_repo import AssessmentRepository


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan context manager for startup and shutdown events."""
    # Startup: Ensure MongoDB connection and indexes
    try:
        await db_manager.ping()
        repo = AssessmentRepository(manager=db_manager)
        await repo.ensure_indexes()
    except Exception:
        # Allow app startup even if DB is connecting asynchronously
        pass

    yield

    # Shutdown: Close MongoDB client connection cleanly
    db_manager.close()


app = FastAPI(
    title=settings.APP_NAME,
    description="Voice/Note to Structured Clinical Assessment Form Filler API Pipeline.",
    version="1.0.0",
    lifespan=lifespan,
)

# Register routers
app.include_router(assessments_router)


@app.get(
    "/health",
    tags=["Infrastructure"],
    status_code=status.HTTP_200_OK,
    summary="Health check endpoint",
)
async def health_check() -> Dict[str, str]:
    """Check application and database health status."""
    try:
        db_ok = await db_manager.ping()
        db_status = "connected" if db_ok else "unhealthy"
    except Exception:
        db_status = "disconnected"

    return {
        "status": "healthy",
        "database": db_status,
        "environment": settings.ENV,
    }
