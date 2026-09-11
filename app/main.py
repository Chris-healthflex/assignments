import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.database.mongodb import init_db, close_db
from app.api.routes.assessments import router as assessments_router

# Setup logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing application components...")
    # Initialize DB (Atlas connection pinged, indexes created)
    connected = init_db()
    if not connected:
        logger.warning("MongoDB Atlas connection could not be established during startup. App will boot, but DB-dependent routes will return 503.")
    yield
    # Cleanup DB connection
    close_db()
    logger.info("Application components cleaned up.")

app = FastAPI(
    title="Clinical Assessment Voice Form Filler API",
    description="WAV Audio transcription and structured clinical data extraction API.",
    version="1.0.0",
    lifespan=lifespan
)

# Exception handler for DB connection issues (503)
@app.exception_handler(ConnectionError)
async def db_connection_exception_handler(request: Request, exc: ConnectionError):
    logger.error(f"Database connection error: {exc}")
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"detail": "Database service is currently unavailable. Please try again later."}
    )

# Exception handler for invalid argument / ObjectId format (400)
@app.exception_handler(ValueError)
async def value_error_exception_handler(request: Request, exc: ValueError):
    # e.g., raised on invalid ObjectId format
    logger.warning(f"Invalid request input value: {exc}")
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"detail": str(exc)}
    )

# Override HTTPExceptions to keep them clean
@app.exception_handler(StarletteHTTPException)
async def custom_http_exception_handler(request: Request, exc: StarletteHTTPException):
    logger.warning(f"HTTP error {exc.status_code}: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )

# Request validation exception handler (422)
@app.exception_handler(RequestValidationError)
async def request_validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.warning(f"Request validation failure: {exc}")
    # Return standard clean validation format
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": exc.errors()}
    )

# Catch-all exception handler (500)
@app.exception_handler(Exception)
async def catch_all_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled system error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An internal server error occurred."}
    )

# Register routes
app.include_router(assessments_router)

@app.get("/health", tags=["Health"])
def health_check():
    from app.database.mongodb import check_db_health
    db_ok = check_db_health()
    return {
        "status": "ok" if db_ok else "degraded",
        "database": "connected" if db_ok else "disconnected"
    }
