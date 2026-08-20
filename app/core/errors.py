from typing import Any
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from app.core.logging import logger


class PipelineBaseError(Exception):
    """Base exception for all pipeline errors."""
    def __init__(self, message: str, status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR, details: Any = None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.details = details or {}


class AudioValidationError(PipelineBaseError):
    """Raised when uploaded audio file fails validation checks."""
    def __init__(self, message: str, details: Any = None):
        super().__init__(message=message, status_code=status.HTTP_400_BAD_REQUEST, details=details)


class TranscriptionError(PipelineBaseError):
    """Raised when audio transcription fails or yields invalid output."""
    def __init__(self, message: str, details: Any = None):
        super().__init__(message=message, status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, details=details)


class ExtractionError(PipelineBaseError):
    """Raised when clinical extraction fails or confidence is below threshold."""
    def __init__(self, message: str, details: Any = None):
        super().__init__(message=message, status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, details=details)


class DatabaseError(PipelineBaseError):
    """Raised when database operations fail."""
    def __init__(self, message: str, details: Any = None):
        super().__init__(message=message, status_code=status.HTTP_503_SERVICE_UNAVAILABLE, details=details)


class AssessmentNotFoundError(PipelineBaseError):
    """Raised when an assessment cannot be found by ID."""
    def __init__(self, assessment_id: str):
        super().__init__(
            message=f"Assessment with ID '{assessment_id}' was not found.",
            status_code=status.HTTP_404_NOT_FOUND,
            details={"assessment_id": assessment_id}
        )


def register_exception_handlers(app: FastAPI) -> None:
    """Registers standard exception handlers on the FastAPI application."""

    @app.exception_handler(PipelineBaseError)
    async def pipeline_error_handler(request: Request, exc: PipelineBaseError) -> JSONResponse:
        logger.error("Pipeline error handled: %s (Status: %s)", exc.message, exc.status_code)
        content = {"detail": exc.message}
        if exc.details:
            content["errors"] = exc.details
        return JSONResponse(status_code=exc.status_code, content=content)

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        logger.warning("Request validation error: %s", exc.errors())
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={"detail": "Request payload validation failed", "errors": exc.errors()}
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled server error: %s", str(exc))
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "An unexpected internal server error occurred."}
        )
