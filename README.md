# Voice/Note → Structured Clinical Assessment

## Overview

Upload a WAV recording of a clinician's assessment note and get back a
structured `FirstAssessment` JSON document that can be saved to and read back
from MongoDB.

The audio is transcribed with OpenAI Whisper, a LangGraph workflow extracts the
clinical information from the transcript, and Pydantic v2 validates the result
against the exact required schema. Nothing clinical is invented: anything the
transcript does not state comes back empty and is reported as unextracted, and an
extraction that is not well supported by the transcript is rejected with HTTP 422
instead of being returned as a guess.

## Architecture

```
WAV upload
  → Whisper transcription        (local openai-whisper, or the OpenAI API)
  → LangGraph extraction         (extract → verify, both structured LLM calls)
  → Pydantic FirstAssessment     (strict validation, exact field names)
  → MongoDB                      (save, get by id, list with date filter)
```

The LangGraph workflow has two nodes. `extract` pulls the clinical information
out of the transcript into a draft where every field is optional. `verify` audits
that draft against the transcript and returns a confidence score plus the paths
of any values it could not find. The confidence is what the 422 threshold checks.

## Tech Stack

| Layer | Choice |
| --- | --- |
| Language | Python 3.11 (3.10+ works) |
| API | FastAPI + Uvicorn |
| Transcription | OpenAI Whisper (`openai-whisper` locally, or the OpenAI API) |
| Extraction | LangGraph + LangChain, Gemini via `langchain-google-genai` |
| Schema | Pydantic v2, settings via `pydantic-settings` |
| Database | MongoDB via Motor (async) |
| Tests | pytest, pytest-asyncio, httpx |

## Project Structure

```
app/
  main.py            FastAPI app, startup index, error handlers
  routes.py          the four endpoints, WAV upload handling
  models.py          FirstAssessment schema + request/response models
  transcription.py   WAV decoding and the Whisper backends
  extraction.py      draft models, LangGraph workflow, mapping to the schema
  prompts.py         extraction and verification prompts
  database.py        Mongo client and the assessment store
  config.py          settings read from the environment
  errors.py          PipelineError, carrying a status code and field details
scripts/
  run_pipeline.py    run a WAV end to end and print the JSON
tests/               schema, transcription, extraction, API and database tests
clinical_assessment.wav
```

## Setup

### Prerequisites

- Python 3.10+
- MongoDB Community Server running locally (Compass alone is not enough, it is
  only a GUI client)
- A Google AI Studio API key for the extraction workflow

On Windows, install MongoDB Community Server from
https://www.mongodb.com/try/download/community, choose the Complete setup and
leave "Install MongoDB as a Service" checked. Check it with `Get-Service MongoDB`.

### Installation

```powershell
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate

pip install -r requirements-dev.txt
copy .env.example .env          # macOS/Linux: cp .env.example .env
```

`requirements.txt` includes `openai-whisper`, which pulls in torch. You can drop
it if you set `WHISPER_BACKEND=openai` and transcribe through the OpenAI API.

### Environment Variables

Copy `.env.example` to `.env` and fill in your key. Only `GOOGLE_API_KEY` has no
usable default.

```
GOOGLE_API_KEY=your-google-api-key
OPENAI_API_KEY=

WHISPER_BACKEND=local
WHISPER_MODEL=small
WHISPER_LANGUAGE=en

LLM_MODEL=gemini-2.5-flash
EXTRACTION_CONFIDENCE_THRESHOLD=0.6

MONGODB_URI=mongodb://localhost:27017
MONGODB_DATABASE=clinical
MONGODB_COLLECTION=assessments

MAX_UPLOAD_BYTES=52428800
LOG_LEVEL=INFO
```

| Variable | Default | Notes |
| --- | --- | --- |
| `GOOGLE_API_KEY` | – | Required. Used by the extraction workflow |
| `OPENAI_API_KEY` | – | Only needed for `WHISPER_BACKEND=openai` |
| `WHISPER_BACKEND` | `local` | `local` runs Whisper in-process, `openai` calls the API |
| `WHISPER_MODEL` | `small` | Local model size, `tiny` through `large` |
| `WHISPER_LANGUAGE` | `en` | Forced transcription language |
| `LLM_MODEL` | `gemini-2.5-flash` | Used for both graph nodes |
| `EXTRACTION_CONFIDENCE_THRESHOLD` | `0.6` | Below this, `/assessments/parse` returns 422 |
| `MONGODB_URI` | `mongodb://localhost:27017` | Connection string |
| `MONGODB_DATABASE` | `clinical` | Created on first insert |
| `MONGODB_COLLECTION` | `assessments` | Created on first insert |
| `MAX_UPLOAD_BYTES` | `52428800` | 50 MB upload limit |
| `LOG_LEVEL` | `INFO` | Root log level |

## Run the API

```powershell
uvicorn app.main:app --reload
```

Interactive docs at http://127.0.0.1:8000/docs.

## API Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| POST | `/assessments/parse` | WAV upload → transcribe → extract → validate → `FirstAssessment` JSON |
| POST | `/assessments` | Save a parsed result to MongoDB |
| GET | `/assessments/{id}` | Fetch one saved assessment |
| GET | `/assessments` | List saved assessments, filtered by date |

**POST /assessments/parse** takes a multipart WAV upload. The response body is
exactly the `FirstAssessment` schema, with no wrapper and no extra fields, so the
tracking information travels in two headers: `X-Extraction-Confidence` and
`X-Unextracted-Fields` (dotted paths of the fields the transcript did not state).

```bash
curl -X POST http://127.0.0.1:8000/assessments/parse \
  -F "file=@clinical_assessment.wav;type=audio/wav"
```

**POST /assessments** saves a parsed result and returns 201 with the stored
record. `metadata` is optional and is stored beside the assessment, never inside
it.

```bash
curl -X POST http://127.0.0.1:8000/assessments \
  -H "Content-Type: application/json" \
  -d '{"assessment": <FirstAssessment JSON>, "metadata": {"sourceFile": "clinical_assessment.wav"}}'
```

**GET /assessments/{id}** returns `{id, createdAt, assessment, metadata}`, or 404
for an unknown or malformed id.

**GET /assessments** lists saved assessments, newest first. `from_date` and
`to_date` are UTC calendar dates matched against `createdAt`, and both bounds are
inclusive.

```bash
curl "http://127.0.0.1:8000/assessments?from_date=2026-08-01&to_date=2026-08-31&limit=20&skip=0"
```

## Run the Pipeline

Runs the whole pipeline on the provided recording and prints the transcript, the
confidence, the unextracted field paths, and the final JSON.

```powershell
python scripts/run_pipeline.py
python scripts/run_pipeline.py path\to\another.wav
```

## Run Tests

```powershell
pytest
```

The extraction and API tests use a stub chat model, so they need no API key and
make no network calls. The database tests skip themselves if `MONGODB_URI` is
unreachable.

## Design Decisions

**Whisper** is used because the assignment requires it. Whisper's own loader
shells out to ffmpeg, and since this endpoint only accepts WAV, `transcription.py`
decodes the file with the standard library `wave` module and numpy instead
(downmix to mono, normalise to float32, resample to 16 kHz). That removes ffmpeg
as a system dependency and catches a broken upload before a model is loaded.

**LangGraph** keeps the two steps explicit and separately debuggable. Grading the
extraction in a second call is more trustworthy than asking the extracting call
to score its own work, and it produces the confidence number the 422 threshold
needs. The assignment fixes the framework but not the model vendor, so extraction
runs on Gemini; only `build_llm` in `extraction.py` knows which provider is used.

**Pydantic v2** enforces the output contract rather than trusting the model.
Every model in the tree sets `extra="forbid"`, every string field defaults to
`""`, and a validator turns `None` into `""`. Array sections default to `[]` and
coerce a single object into a one-element list. The result cannot contain an extra
or renamed field, a null string, or a bare object where an array belongs.

**Uncertain clinical information** is never filled in. The LLM extracts into a
draft where every field is `Optional[str]`, which gives it a way to say "not
stated" that is distinct from an empty string, and the prompt forbids inferring
values or using placeholders like `"N/A"`. Fields the transcript does not mention
stay empty and are listed in `X-Unextracted-Fields`. If the verify node scores
the extraction below the threshold, the request fails with 422 and the response
names the ungrounded fields.

**MongoDB** suits documents whose shape is already defined by the schema, so an
assessment can be stored and read back without translation. A record is
`{_id, createdAt, assessment, metadata}`: the assessment is stored exactly as
validated, provenance lives in `metadata`, and `createdAt` is indexed for the
date filter.

## Error Handling

Every failure raises a `PipelineError` with a code and field-level `details`, and
one handler renders it as `{code, message, details}`. FastAPI's own request
validation errors are reshaped into the same envelope.

| Status | Code | When |
| --- | --- | --- |
| 400 | `invalid_audio` | Not a WAV, unreadable, empty, or over the size limit |
| 422 | `low_extraction_confidence` | Confidence below the threshold. `details` carries the score, the threshold, the verifier's reason, and one entry per ungrounded field |
| 422 | `request_validation_failed` | Payload does not match the schema, with the offending field path |
| 422 | `extraction_failed` | Transcript too short to extract from |
| 502 | `transcription_failed` | Whisper failed or returned nothing |
| 502 | `extraction_failed` | A model call failed |
| 503 | `database_unavailable` | MongoDB unreachable |
| 404 | `assessment_not_found` | Unknown or malformed id |

## Notes / Assumptions

- The transcript is the only source of truth. Unstated fields come back as `""`
  or `[]`, never as `"N/A"` or a plausible default.
- Timeframes stay as spoken ("eight months", "in six weeks"). Converting them to
  calendar dates would mean inventing information the recording does not carry.
- All `FirstAssessment` leaf values are strings, including measurements like
  `"124"`, because the schema specifies string fields; the unit goes in
  `unitName`.
- `left` and `right` are filled only when the transcript states a side, otherwise
  the measurement goes in `value`.
- `POST /assessments` re-validates its payload but does not re-run extraction; it
  trusts the caller to send a previously parsed result.
- The list endpoint filters on `createdAt`, the time the assessment was saved,
  because the recordings carry no encounter date of their own.
- Confidence measures how well the extraction is grounded in the transcript, not
  clinical quality.
- Whisper can mis-hear clinical terminology, so the prompt tells the model to copy
  what the transcript says rather than correct it. Reviewing the transcript stays
  part of the workflow, which is why the pipeline script prints it.
