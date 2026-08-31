import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.database import close_client, get_collection
from app.errors import PipelineError
from app.models import ErrorResponse
from app.routes import router

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logging.basicConfig(
        level=settings.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    try:
        await get_collection(settings).create_index("createdAt")
    except Exception as exc:
        logger.warning("could not reach mongodb at startup: %s", exc)
    yield
    close_client()


app = FastAPI(title="Clinical Assessment Pipeline", version="1.0.0", lifespan=lifespan)
app.include_router(router)


@app.exception_handler(PipelineError)
async def handle_pipeline_error(request: Request, exc: PipelineError) -> JSONResponse:
    logger.warning("%s on %s: %s", exc.code, request.url.path, exc.message)
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            code=exc.code, message=exc.message, details=exc.details
        ).model_dump(),
    )


@app.exception_handler(RequestValidationError)
async def handle_validation_error(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    details = [
        {
            "field": ".".join(str(part) for part in error["loc"][1:]) or "body",
            "message": error["msg"],
            "type": error["type"],
        }
        for error in exc.errors()
    ]
    return JSONResponse(
        status_code=422,
        content=ErrorResponse(
            code="request_validation_failed",
            message="The request payload does not match the required schema.",
            details=details,
        ).model_dump(),
    )
