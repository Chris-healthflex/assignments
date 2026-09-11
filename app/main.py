from fastapi import FastAPI

from app.api.assessments import router as assessments_router


app = FastAPI(
    title="Stance Health Clinical Assessment API",
    version="1.0.0",
    description="Voice/Note to Structured Clinical Assessment Form Filler",
)

app.include_router(assessments_router)


@app.get("/health")
def health_check():
    return {"status": "ok"}