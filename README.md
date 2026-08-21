# Clinical Assessment Pipeline

Python 3.10+ service that converts a clinician-patient WAV recording into a strict `FirstAssessment` JSON document and stores validated assessments in MongoDB.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[test]"
Copy-Item .env.example .env
```

Set `OPENAI_API_KEY` in `.env` for the LangGraph extraction model. Set `MONGODB_URI` if MongoDB is not running at the default local URI. Whisper downloads the configured local model on first use and may require `ffmpeg` on the host.

## Run

Start the API:

```powershell
uvicorn app.main:app --reload
```

Run the supplied WAV through the complete pipeline and print the JSON:

```powershell
python tests/run_pipeline.py clinical_assessment.wav
```

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

The tests cover strict schema keys, array defaults, confidence rejection, and WAV input validation. Endpoint and MongoDB integration tests require the optional runtime dependencies and a test database.

## Design Decisions

- Pydantic models reject unknown fields so response shape cannot drift silently.
- Whisper and LangGraph are lazy-loaded at execution time, keeping API import and schema tests independent of model downloads.
- Uploaded audio is written to a temporary file and removed after parsing.
- MongoDB stores the validated assessment payload plus internal `created_at` and `_id` metadata; API records expose the identifier as `id`.
- The nested fields implemented here are the fields explicitly listed in the assignment brief. If the repository supplies a fuller production `FirstAssessment` contract, update only `app/models/first_assessment.py` and its schema tests to that authoritative definition.
