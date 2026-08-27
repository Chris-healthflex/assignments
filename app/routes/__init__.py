"""API routes, aggregated into a single router for the app to include.

`parse` is registered before `assessments` so the literal `/assessments/parse`
path is matched ahead of the `/assessments/{id}` pattern.
"""

from fastapi import APIRouter

from app.routes import assessments, health, parse

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(parse.router)
api_router.include_router(assessments.router)

__all__ = ["api_router"]
