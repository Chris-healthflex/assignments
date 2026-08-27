import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import settings
from app.routers.assessments import router as assessments_router
from app.services.database import db

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("clinical_pipeline")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize DB connection on startup
    logger.info("Starting Clinical Audio Assessment Pipeline service...")
    db.connect()
    yield
    logger.info("Shutting down Clinical Audio Assessment Pipeline service...")


app = FastAPI(
    title=settings.APP_NAME,
    version="1.0.0",
    description=(
        "Production backend pipeline that transcribes clinical audio recordings via Whisper, "
        "extracts clinical assessment data via LangGraph/LangChain into the strict FirstAssessment schema, "
        "and persists data in MongoDB."
    ),
    lifespan=lifespan,
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers
app.include_router(assessments_router)


@app.get("/", tags=["Health"])
async def root():
    return {
        "status": "healthy",
        "service": settings.APP_NAME,
        "database_connected": not db.is_mock,
        "database_mode": "mongodb" if not db.is_mock else "in-memory-mock",
        "whisper_mode": settings.WHISPER_MODE,
        "docs_url": "/docs",
    }


@app.get("/health", tags=["Health"])
async def health_check():
    return {
        "status": "ok",
        "database_connected": not db.is_mock,
    }
