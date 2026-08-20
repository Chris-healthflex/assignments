from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.core.config import get_settings
from app.core.logging import setup_logging, logger
from app.core.errors import register_exception_handlers
from app.database.mongodb import db_manager
from app.api.routes.assessments import router as assessments_router


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manages application startup and shutdown lifecycles."""
    setup_logging()
    settings = get_settings()
    logger.info("Starting %s (v%s) in [%s] mode", settings.APP_NAME, settings.APP_VERSION, settings.ENVIRONMENT)

    # Initialize Database connection
    try:
        db_manager.connect()
    except Exception as exc:
        logger.warning("MongoDB connection could not be established at startup: %s", exc)

    yield

    # Shutdown
    logger.info("Shutting down application...")
    db_manager.close()


def create_application() -> FastAPI:
    """Factory creating and configuring the FastAPI application instance."""
    settings = get_settings()

    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description="Clinical Audio -> Structured FirstAssessment Backend Pipeline (FastAPI + Whisper + LangGraph + Pydantic v2 + MongoDB)",
        docs_url="/docs",

        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # CORS configuration
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register Exception Handlers
    register_exception_handlers(app)

    # Health check endpoint
    @app.get("/health", tags=["Health"], summary="Health check endpoint")
    async def health_check() -> dict[str, str]:
        return {
            "status": "healthy",
            "version": settings.APP_VERSION,
            "environment": settings.ENVIRONMENT,
        }

    # Serve static UI
    static_dir = Path(__file__).parent / "static"
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.get("/", include_in_schema=False)
    async def serve_ui() -> FileResponse:
        return FileResponse(str(static_dir / "index.html"))

    # Register API Routers
    app.include_router(assessments_router)

    return app


app = create_application()
