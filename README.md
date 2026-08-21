# Clinical Assessment Pipeline

Python 3.10+ service that converts a clinician-patient WAV recording into a strict `FirstAssessment` JSON document and stores validated assessments in MongoDB.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[test]"
Copy-Item .env.example .env
```

Set `GROQ_API_KEY` in `.env` for the LangGraph extraction model. The implementation originally targeted OpenAI `gpt-4o-mini`, but that path was blocked by quota. It migrated to Groq `openai/gpt-oss-120b` after Groq deprecated `llama-3.3-70b-versatile` in August 2026. Set `MONGODB_URI` if MongoDB is not running at the default local URI.

Whisper runs locally with the `base` model by default. This keeps audio processing independent of a second hosted transcription API and is a reasonable speed/accuracy choice for the assignment. The model downloads on first use and may require `ffmpeg` on the host.

## Run

Start the API:

```powershell
uvicorn app.main:app --reload
```

Run the supplied WAV through the complete pipeline and print the JSON:

```powershell
python tests/run_pipeline.py clinical_assessment.wav
```

The interactive API documentation is available at `http://127.0.0.1:8000/docs` after the server starts. For local MongoDB, run `docker run -d -p 27017:27017 --name clinical-mongo mongo:7`.

## Endpoints

- `POST /assessments/parse`: multipart form upload named `file`; accepts WAV and returns `FirstAssessment` JSON.
- `POST /assessments`: validates and saves a `FirstAssessment` document.
- `GET /assessments/{id}`: retrieves one saved assessment.
- `GET /assessments?from_date=...&to_date=...`: lists assessments with optional creation-date bounds.

Extraction below `CONFIDENCE_THRESHOLD` returns HTTP 422 with field-level details. The pipeline does not infer clinical values, scores, measurements, dates, diagnoses, or recommendations that are absent from the transcript.

## Tests

```powershell
pytest
```

The tests use mocked Whisper, Groq/LangGraph, and `mongomock`, so they do not require a live model API, audio model download, or MongoDB connection. They cover strict schema keys, hallucination guards, confidence rejection, five-item array preservation, all endpoint success/failure paths, date filtering errors, and persistence.

## Design Decisions

- Pydantic models reject unknown fields so response shape cannot drift silently.
- Whisper and LangGraph are lazy-loaded at execution time, keeping API import and schema tests independent of model downloads.
- Uploaded audio is written to a temporary file and removed after parsing.
- MongoDB stores the validated assessment payload plus internal `created_at` and `_id` metadata; API records expose the identifier as `id`.
- The confidence threshold is `0.75`. It is applied to self-reported, per-field confidence scores returned by the extraction model; fields below that score are returned as HTTP 422 for human review.
- The Pydantic model matches the supplied production schema image: all seven sections, nested key names, array shapes, and non-null string leaf fields are represented exactly. No frontend, static assets, HTML, or unrelated clinical features are included.
