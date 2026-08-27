"""Service metadata endpoint."""

from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/")
async def root():
    return {
        "status": "online",
        "service": "Clinical Assessment Audio-to-JSON Pipeline",
        "endpoints": [
            "POST /assessments/parse",
            "POST /assessments",
            "GET /assessments/{id}",
            "GET /assessments"
        ]
    }
