from fastapi import FastAPI

from app.routers import assessments

app = FastAPI(
    title="Stance Health - Clinical Assessment Pipeline",
    description="WAV clinical session -> transcription -> structured FirstAssessment JSON",
    version="1.0.0",
)

app.include_router(assessments.router)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}
