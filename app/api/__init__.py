"""API package for FastAPI routers and dependency injection."""

from app.api.assessments import router as assessments_router

__all__ = ["assessments_router"]
