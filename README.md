# Voice/Note → Structured Clinical Assessment Form Filler

A FastAPI service that converts a clinician-patient WAV recording into a structured clinical assessment using local Whisper transcription, LangGraph-based extraction, Pydantic validation, and MongoDB persistence.

## Tech Stack

- Python 3.10+
- FastAPI
- Faster-Whisper for local transcription
- LangGraph
- Pydantic v2
- MongoDB Atlas
- PyMongo
- Pytest

## Project Structure

```text
app/
├── api/
│   └── assessments.py
├── core/
│   └── config.py
├── models/
│   └── schemas.py
├── services/
│   ├── database.py
│   ├── extraction.py
│   └── transcription.py
└── main.py

tests/
├── test_config.py
├── test_extraction.py
├── test_extraction_unit.py
├── test_schema.py
└── test_transcription.py
```

## Setup

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```env
MONGODB_URI=<your-mongodb-atlas-connection-string>
```

The `.env` file is intentionally excluded from Git.

## Run the API

```powershell
python -m uvicorn app.main:app --reload
```

Open Swagger:

```text
http://127.0.0.1:8000/docs
```

## API Endpoints

### Health Check

```http
GET /health
```

### Parse Assessment

```http
POST /assessments/parse
```

Upload a WAV file using the `file` form field.

The endpoint:

1. Saves the uploaded WAV temporarily.
2. Transcribes it locally using Faster-Whisper.
3. Extracts structured clinical information.
4. Validates the result using Pydantic.
5. Rejects low-confidence extraction with HTTP 422.
6. Saves the assessment to MongoDB.
7. Returns the saved assessment including its MongoDB ID.

### Get Assessments

```http
GET /assessments/
```

Supports optional date filtering using `date_from` and `date_to`.

### Get Assessment by ID

```http
GET /assessments/{assessment_id}
```

Returns HTTP 404 when the assessment does not exist.

## Validation and Safety

The extraction pipeline is designed to avoid inventing clinical information.

- Clinical values are only populated when supported by the transcript.
- Missing or insufficiently extracted required information is reported through the confidence validation flow.
- Low-confidence extraction returns HTTP 422 with field-level information.
- Array fields remain arrays even when only one item is present.
- String fields are represented as strings rather than `null`.

## Testing

Run the complete test suite:

```powershell
python -m pytest -v
```

The test suite covers:

- Configuration
- Clinical assessment extraction
- Clinical detail extraction
- Objective measurement extraction
- Pydantic schema validation
- Local transcription
- Low-confidence HTTP 422 behavior

## Sample Audio

`clinical_assessment.wav` is included as the sample input used by the automated extraction pipeline tests.

## MongoDB

The application stores assessments in:

```text
Database: clinical_assessment
Collection: assessments
```

MongoDB connection credentials are loaded from the local `.env` file and are never committed to the repository.