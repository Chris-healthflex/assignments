"""FastAPI application entry point.

Creates the app and wires in the routers. All endpoint logic lives in
`app.routes`, all pipeline work in `app.services`.

Run with:  uvicorn app.main:app --reload --port 8000
"""

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


from app.routes import api_router

logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="Clinical Assessment Extraction API",
    description=(
        "FastAPI service to transcribe clinical audio sessions, extract structured "
        "FirstAssessment JSON via a LangGraph agent, and store/retrieve in MongoDB."
    ),
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)
