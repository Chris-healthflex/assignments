from fastapi import FastAPI
from app.api.assessments import router as assessments_router

app = FastAPI(title="Stance Health - Clinical Assessment Pipeline")

app.include_router(assessments_router)


@app.get("/")
async def root():
    return {"status": "ok", "service": "clinical-assessment-pipeline"}