"""HTTP routes — the 4 endpoints.

EP1  POST /transcribe-assess : upload a WAV -> run full pipeline -> JSON (optional save)
EP2  POST /assessments       : save a parsed result to MongoDB
EP3  GET  /assessments/{id}  : retrieve a saved assessment by id
EP4  GET  /assessments       : list all, filterable by date range
"""
from __future__ import annotations

import os
import tempfile

from fastapi import APIRouter, File, HTTPException, Query, UploadFile

from app.api.dependencies import parse_date
from app.api.schemas import AssessmentCreate, AssessmentList
from app.services import assessment_service as svc

router = APIRouter()


@router.post("/transcribe-assess", summary="Audio -> structured assessment (EP1)")
async def transcribe_assess(
    file: UploadFile = File(...),
    save: bool = Query(False, description="Persist the result to MongoDB"),
):
    if not file.filename or not file.filename.lower().endswith(".wav"):
        raise HTTPException(status_code=422, detail="Please upload a .wav file")

    suffix = os.path.splitext(file.filename)[1] or ".wav"
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name
        try:
            return await svc.process_and_optionally_save(tmp_path, save=save)
        except ValueError as exc:  # audio validation errors
            raise HTTPException(status_code=422, detail=str(exc))
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


@router.post("/assessments", status_code=201, summary="Save a parsed result (EP2)")
async def create_assessment(body: AssessmentCreate):
    payload = body.model_dump(mode="json")
    new_id = await svc.save_result(payload)
    return {"id": new_id}


@router.get("/assessments/{assessment_id}", summary="Retrieve by id (EP3)")
async def read_assessment(assessment_id: str):
    stored = await svc.get_assessment(assessment_id)
    if stored is None:
        raise HTTPException(status_code=404, detail="Assessment not found")
    return stored


@router.get("/assessments", response_model=AssessmentList, summary="List / filter by date (EP4)")
async def list_assessments(
    start_date: str | None = Query(None, description="ISO date lower bound (inclusive)"),
    end_date: str | None = Query(None, description="ISO date upper bound (inclusive)"),
):
    start = parse_date(start_date, "start_date")
    end = parse_date(end_date, "end_date")
    items = await svc.list_assessments(start, end)
    return AssessmentList(count=len(items), items=items)
