from fastapi import FastAPI

from app.api.assessments import router as assessments_router


app = FastAPI(
    title="Clinical Assessment API",
    description="Voice/Note to Structured Clinical Assessment API",
    version="1.0.0",
)


app.include_router(assessments_router)


@app.get("/")
def root():
    return {
        "message": "Clinical Assessment API is running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }