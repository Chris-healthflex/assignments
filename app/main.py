from fastapi import FastAPI

from app.api.assessments import router as assessments_router

app = FastAPI(title="Clinical Assessment Pipeline", version="0.1.0")
app.include_router(assessments_router)
