import logging

from fastapi import FastAPI

from app.routers import assessments

logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="Stance Health — Clinical Assessment Pipeline",
    description="Voice/Note -> Structured Clinical Assessment Form Filler",
    version="1.0.0",
)

app.include_router(assessments.router)


@app.get("/health")
async def health():
    return {"status": "ok"}
