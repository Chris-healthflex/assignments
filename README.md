# Clinical Voice → Structured Assessment Pipeline

FastAPI service that transcribes a WAV session, extracts only transcript-supported
clinical information with LangGraph, validates the exact `FirstAssessment` schema,
and stores confirmed results in MongoDB.

## Setup

1. Use Python 3.10+ and create/activate a virtual environment.
2. Run `pip install -r requirements.txt`.
3. Copy `.env.example` to `.env`, then set `OPENAI_API_KEY` and `MONGODB_URI`.
4. Start MongoDB, then run `uvicorn app.main:app --reload`.

Swagger UI is at `http://127.0.0.1:8000/docs`.

## API

- `POST /assessments/parse` accepts multipart field `file` containing a WAV and
  returns only `FirstAssessment` JSON.
- `POST /assessments` stores a validated `FirstAssessment` document.
- `GET /assessments/{id}` returns one saved document.
- `GET /assessments?date=YYYY-MM-DD` lists saved documents, optionally filtered
  by their UTC creation date.

If an extraction is incomplete or ambiguous, parsing returns HTTP 422 with
field-level detail. It never fills uncertain clinical values, dates, or scores.

## Test and sample run

Run schema/API tests with:

```powershell
pytest
```

Run the complete supplied-WAV pipeline and print the schema JSON with:

```powershell
python tests/test_pipeline.py C:\Users\akjee\Downloads\clinical_assessment.wav
```

The script exits with code 2 and prints field-level issues when the transcript
does not support a complete assessment; this is the intentional safety behavior.

## Design decisions

Whisper uses the OpenAI transcription API (`whisper-1`) for dependable WAV
handling without shipping a local model. A one-node LangGraph makes the
transcription → extraction boundary explicit and is easy to extend with review
or de-identification nodes. The extraction model uses Pydantic structured output
and an internal uncertainty envelope; only the exact production schema is ever
returned from the parse endpoint. MongoDB documents add `_id` and `createdAt`
only at persistence time, while the assessment payload itself remains unchanged.
