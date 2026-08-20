"""FastAPI service: WAV -> transcript -> grounded FirstAssessment -> MongoDB.

Four endpoints, matching the brief:

    POST /assessments/parse        multipart WAV  -> draft assessment, or 422
    POST /assessments              draft          -> saved, with an id
    GET  /assessments/{id}         id             -> one assessment
    GET  /assessments?date=        YYYY-MM-DD     -> that day's assessments

Every one of them speaks ``StoredAssessment``: the untouched ``FirstAssessment``
contract under ``assessment``, with transcript, confidence and identifiers
wrapped around it. A wrapper is unavoidable -- the brief forbids extra fields
inside the contract *and* requires confidence to come back with the result, so
the confidence has to live somewhere outside. Given that, one envelope used
identically by all four endpoints beats four different response shapes.
"""

from __future__ import annotations

import asyncio
import logging
import re
import tempfile
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from pymongo.errors import PyMongoError

from app import db
from app.config import get_settings
from app.extraction import ExtractionFailed, ExtractionUnavailable, extract
from app.schemas import FieldEvidence, StoredAssessment
from app.transcription import TranscriptionError, transcribe

settings = get_settings()
logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)

# faster-whisper holds one model in memory and is not safe to call from several
# threads at once. Transcription is also CPU-bound, so running two at a time
# would not be faster anyway -- requests queue here rather than corrupt state.
# Extraction is deliberately *not* serialised: it is network-bound and already
# paced by the rate limiter in `extraction._llm`.
_TRANSCRIBE_LOCK = asyncio.Semaphore(1)

# faster-whisper decodes through the PyAV bindings it ships with, so mp3, m4a,
# flac and ogg need no separate ffmpeg install. WAV, MP3 and M4A were each
# checked against the same source recording and produced identical transcripts.
AUDIO_SUFFIXES = frozenset({".wav", ".wave", ".mp3", ".m4a", ".flac", ".ogg", ".webm"})


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        await db.ensure_indexes()
    except PyMongoError:
        # Start anyway. A service that refuses to boot because the database is
        # briefly unreachable cannot even serve /health to say so.
        logger.warning("Could not create indexes at startup", exc_info=True)
    yield
    await db.close()


app = FastAPI(
    title=settings.app_name,
    version="1.0.0",
    summary="Clinical first-assessment extraction from consultation audio.",
    lifespan=lifespan,
)


# --------------------------------------------------------------------------- #
# The 422 body
# --------------------------------------------------------------------------- #
_INDEX = re.compile(r"\[(\d+)\]")


class FieldError(BaseModel):
    """One low-confidence field, shaped like a FastAPI validation error.

    Deliberately the same `loc`/`msg`/`type`/`ctx` shape FastAPI uses for its own
    422s, so a client that already knows how to render a validation error can
    render these with no extra code.
    """

    loc: list[str | int] = Field(description="Path to the field inside the response.")
    msg: str
    type: str
    ctx: dict[str, Any] = Field(default_factory=dict)


class LowConfidence(BaseModel):
    """The 422 payload: what is wrong, plus the draft it is wrong in."""

    detail: list[FieldError]
    transcript: str = ""
    flags: dict[str, Any] = Field(default_factory=dict)
    assessment: dict[str, Any] = Field(default_factory=dict)


def _loc(path: str) -> list[str | int]:
    """``objectiveAssessment.tests[1].left`` -> ``[..., 'tests', 1, 'left']``.

    Prefixed with ``assessment`` so the path is valid against the response body
    the client is holding, not against the bare contract.
    """
    parts: list[str | int] = ["assessment"]
    for token in path.split("."):
        name = _INDEX.sub("", token)
        if name:
            parts.append(name)
        parts.extend(int(i) for i in _INDEX.findall(token))
    return parts


def _section_error(section: str) -> FieldError:
    """A section that is empty because the call producing it failed.

    Reported in the same list as the low-confidence fields, because from the
    caller's side it is the same kind of problem: part of this document cannot
    be trusted. Leaving it out would be worse than useless -- an empty section
    is a perfectly ordinary, correct answer when the clinician did not mention
    it, so with no error attached there is nothing to tell the two apart.
    """
    return FieldError(
        loc=["assessment", section],
        msg=(
            "This section could not be extracted: the model call for it failed "
            "after retries. It is empty because we could not ask, not because "
            "the recording was silent."
        ),
        type="section_unavailable",
        ctx={"section": section},
    )


def _field_error(evidence: FieldEvidence, threshold: float) -> FieldError:
    """Turn one piece of evidence into an error a human can act on.

    The message says *why* rather than just "low confidence": a value nobody can
    trace to the recording is a different problem from one the microphone
    garbled, and they call for different fixes.
    """
    if not evidence.evidenceFound:
        kind = "unverified_evidence"
        default = "No part of the transcript supports this value."
    else:
        kind = "low_confidence"
        default = (
            f"Confidence {evidence.confidence:.0%} is below the {threshold:.0%} "
            "needed to return this value without review."
        )
    return FieldError(
        loc=_loc(evidence.field),
        msg=evidence.reason or default,
        type=kind,
        ctx={
            "value": evidence.value,
            "evidence": evidence.evidence,
            "confidence": round(evidence.confidence, 4),
            "modelConfidence": evidence.modelConfidence,
            "audioConfidence": evidence.audioConfidence,
            "contextConfidence": evidence.contextConfidence,
        },
    )


# --------------------------------------------------------------------------- #
# Upload handling
# --------------------------------------------------------------------------- #
async def _spool(file: UploadFile, limit_bytes: int) -> Path:
    """Stream an upload to a temp file, refusing anything oversized.

    Streamed in chunks rather than read whole: an unbounded ``read()`` lets one
    request decide how much memory the process uses. The size is checked as it
    arrives, so an oversized file is rejected partway through rather than after
    it has all been written to disk.
    """
    name = Path(file.filename or "audio.wav").name  # strip any directory games
    suffix = Path(name).suffix.lower() or ".wav"
    if suffix not in AUDIO_SUFFIXES:
        raise HTTPException(
            status_code=415,
            detail=(
                f"Unsupported audio format {suffix!r}. Accepted: "
                + ", ".join(sorted(AUDIO_SUFFIXES))
            ),
        )

    written = 0
    oversized = False
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp_path = Path(tmp.name)
        while chunk := await file.read(1024 * 1024):
            written += len(chunk)
            if written > limit_bytes:
                oversized = True
                break
            tmp.write(chunk)

    if oversized:
        tmp_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=413,
            detail=f"Audio exceeds the {limit_bytes // (1024 * 1024)} MB limit.",
        )
    if written == 0:
        tmp_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    return tmp_path


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #
@app.get("/health", tags=["ops"])
async def health() -> JSONResponse:
    """Liveness plus database reachability.

    503 when Mongo is down rather than a cheerful 200: a health check that
    always passes tells an orchestrator nothing.
    """
    mongo = await db.ping()
    return JSONResponse(
        status_code=200 if mongo else 503,
        content={"status": "ok" if mongo else "degraded", "mongo": mongo},
    )


# Declared before /assessments/{assessment_id}: FastAPI matches routes in the
# order they are registered, and the dynamic route would otherwise swallow
# "parse" as an id.
@app.post(
    "/assessments/parse",
    response_model=StoredAssessment,
    responses={422: {"model": LowConfidence}},
    tags=["assessments"],
)
async def parse_assessment(file: UploadFile = File(...)):
    """Transcribe a recording and extract a first assessment. Does not save.

    Returns 422 when any field falls below the confidence threshold -- with the
    draft attached. Withholding the draft would be the wrong kind of strict: the
    clinician needs to see what was heard in order to correct it, and the
    flagged fields are precisely the ones they should be looking at.
    """
    audio_path = await _spool(file, settings.max_upload_mb * 1024 * 1024)
    try:
        # Both of these are synchronous and slow -- Whisper for minutes, Gemini
        # for seconds. Calling them directly from an async handler would block
        # the event loop and freeze every other request in the process.
        async with _TRANSCRIBE_LOCK:
            transcription = await run_in_threadpool(transcribe, audio_path)
        result = await run_in_threadpool(extract, transcription.text, transcription)
    except TranscriptionError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except NotImplementedError as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc
    except ExtractionUnavailable as exc:
        # 502, not 422: nothing was wrong with the request. The model provider
        # did not answer, and the useful advice is "try again", not "fix your
        # input". Must be caught before ExtractionFailed, which it subclasses.
        logger.error("Extraction provider unavailable: %s", exc)
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except ExtractionFailed as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    finally:
        audio_path.unlink(missing_ok=True)

    draft = StoredAssessment(
        audioFilename=Path(file.filename or "").name,
        transcript=transcription.text,
        flags=result.flags,
        assessment=result.assessment,
    )

    failing = result.failing()
    missing = list(result.flags.failedSections)

    # A missing section is reported even when every field that *did* come back
    # scored well. Confidence is an average over what was extracted, so losing a
    # section tends to raise it -- which is precisely why the score cannot be
    # the only thing standing between a partial result and a 200.
    if failing or missing:
        threshold = settings.extraction_confidence_threshold
        logger.info(
            "Parsed with %d field(s) below %.2f and %d unavailable section(s)",
            len(failing),
            threshold,
            len(missing),
        )
        body = LowConfidence(
            detail=[_section_error(s) for s in missing]
            + [_field_error(f, threshold) for f in failing],
            transcript=draft.transcript,
            flags=draft.flags.model_dump(mode="json"),
            assessment=draft.assessment.model_dump(mode="json"),
        )
        return JSONResponse(status_code=422, content=body.model_dump(mode="json"))

    return draft


@app.post("/assessments", response_model=StoredAssessment, status_code=201, tags=["assessments"])
async def create_assessment(payload: StoredAssessment) -> StoredAssessment:
    """Persist an assessment and return it with its new id.

    No confidence gate here, on purpose. The gate belongs at the boundary where
    a *machine* produces values; this endpoint receives what a human has already
    reviewed. Re-checking it would mean a clinician who corrected a misheard
    measurement could not save the correction.
    """
    try:
        payload.id = await db.save_assessment(payload)
    except PyMongoError as exc:
        raise _unavailable(exc) from exc
    return payload


@app.get("/assessments", response_model=list[StoredAssessment], tags=["assessments"])
async def list_assessments(
    date_: date | None = Query(
        default=None,
        alias="date",
        description="Calendar day (UTC) to filter on, as YYYY-MM-DD.",
    ),
    limit: int = Query(default=50, ge=1, le=200),
    skip: int = Query(default=0, ge=0),
) -> list[StoredAssessment]:
    """List assessments, newest first, optionally narrowed to a single day.

    ``date`` filters on when the assessment was captured. The contract has no
    date of its own -- its only dates are goal ``targetDate`` values, which are
    targets rather than a record of when anything happened -- so the envelope's
    ``createdAt`` is the only thing "on this date" can honestly mean.
    """
    try:
        return await db.list_assessments(date_, limit=limit, skip=skip)
    except PyMongoError as exc:
        raise _unavailable(exc) from exc


@app.get("/assessments/{assessment_id}", response_model=StoredAssessment, tags=["assessments"])
async def get_assessment(assessment_id: str) -> StoredAssessment:
    try:
        stored = await db.get_assessment(assessment_id)
    except PyMongoError as exc:
        raise _unavailable(exc) from exc
    if stored is None:
        raise HTTPException(status_code=404, detail="No assessment with that id.")
    return stored


def _unavailable(exc: PyMongoError) -> HTTPException:
    """503, not 500: the request was fine, the database was not."""
    logger.error("MongoDB operation failed: %s", exc)
    return HTTPException(status_code=503, detail="The database is unavailable.")


# --------------------------------------------------------------------------- #
# Review UI
# --------------------------------------------------------------------------- #
# One static page, served from the API's own origin. That is the reason there is
# no CORS configuration anywhere in this file: same host, so the browser never
# treats the calls as cross-origin and no permissive header has to be invented.
# Mounted last, after every route, so it can never shadow one.
STATIC_DIR = Path(__file__).parent / "static"


@app.get("/", include_in_schema=False)
async def index() -> RedirectResponse:
    return RedirectResponse("/ui/")


app.mount("/ui", StaticFiles(directory=STATIC_DIR, html=True), name="ui")
