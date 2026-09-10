from fastapi import FastAPI

from app.api.assessments import router as assessments_router


app = FastAPI(
    title="Clinical Assessment Parser",
    version="1.0.0",
)


app.include_router(assessments_router)


@app.get("/health")
def health_check():
    return {"status": "ok"}