# Voice/Note → Structured Clinical Assessment Form Filler

A FastAPI service that converts a clinical WAV recording into a structured `FirstAssessment` using local OpenAI Whisper transcription, LangGraph orchestration, Gemini structured extraction, Pydantic v2 validation, and MongoDB persistence.

## Architecture

WAV audio → Whisper transcription → LangGraph extraction workflow → Gemini structured output → Pydantic `FirstAssessment` → MongoDB

## Tech Stack

- Python 3.10+
- FastAPI
- OpenAI Whisper
- LangChain
- LangGraph
- Gemini
- Pydantic v2
- MongoDB
- PyMongo
- Git

## Project Structure

```text
app/
├── api/
│   └── assessments.py
├── db/
│   └── mongodb.py
├── graph/
│   └── clinical_graph.py
├── models/
│   ├── assessment.py
│   └── confidence.py
├── services/
│   ├── extraction.py
│   └── transcription.py
└── main.py

test_extraction.py
requirements.txt
.env.example
README.md
```

## Setup

### 1. Create and activate a virtual environment

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

### 3. Install FFmpeg

Whisper requires FFmpeg for audio processing.

On Windows:

```powershell
winget install Gyan.FFmpeg
```

Verify:

```powershell
ffmpeg -version
```

### 4. Configure environment variables

Create a local `.env` file using `.env.example` as a template.

Required variables:

```env
GEMINI_API_KEY=your_gemini_api_key
GEMINI_MODEL=gemini-3.6-flash
MONGODB_URI=your_mongodb_connection_string
MONGODB_DATABASE=stance_health
CONFIDENCE_THRESHOLD=0.70
```

Never commit `.env` or API credentials.

## Run the API

```powershell
python -m uvicorn app.main:app --reload
```

API documentation:

```text
http://127.0.0.1:8000/docs
```

Health check:

```text
GET /health
```

## API Endpoints

### Parse a clinical WAV

```text
POST /assessments/parse
```

Upload a `.wav` file using the `file` form field.

The service:

1. Validates the WAV upload.
2. Transcribes audio using local Whisper.
3. Sends the transcript through the LangGraph extraction workflow.
4. Extracts the `FirstAssessment`.
5. Validates the result using Pydantic.
6. Returns the structured assessment.

If extraction confidence falls below the configured threshold, the endpoint returns HTTP `422` with field-level confidence details.

### Save an assessment

```text
POST /assessments
```

Accepts a `FirstAssessment` JSON document and stores it in MongoDB.

### Retrieve an assessment

```text
GET /assessments/{id}
```

Returns a saved assessment by MongoDB ID.

### List assessments

```text
GET /assessments
```

Optional query parameters:

```text
start_date
end_date
```

These can be used to filter assessments by creation date.

## Test Script

The supplied WAV can be processed directly:

```powershell
python test_extraction.py clinical_assessment.wav
```

The script:

* Transcribes the WAV using Whisper.
* Runs the LangGraph extraction pipeline.
* Prints the transcript.
* Prints the resulting `FirstAssessment` JSON.

A different WAV file can also be supplied:

```powershell
python test_extraction.py path\to\audio.wav
```

## Design Decisions

### Local Whisper

Whisper runs locally so audio transcription does not require an external transcription API.

### LangGraph

LangGraph provides an explicit workflow boundary between transcription input and clinical assessment extraction and makes the extraction pipeline extensible.

### Structured Gemini output

The extraction model is constrained to the Pydantic `ExtractionResult`, containing:

* the required `FirstAssessment`
* internal confidence scores

Confidence metadata is used internally for threshold validation and is not included in the final assessment JSON.

### Confidence threshold

Each top-level assessment section receives an internal confidence score.

The configured threshold defaults to:

```text
0.70
```

If any required section falls below the threshold, the API returns HTTP `422` rather than silently returning an unreliable assessment.

### Pydantic validation

`FirstAssessment` uses Pydantic v2 models with `extra="forbid"` to prevent unexpected fields and preserve the required schema.

Arrays remain arrays even when they contain a single item, and string fields use empty strings rather than `null` when information is not available.

### MongoDB

MongoDB stores the structured assessment together with its creation timestamp, allowing retrieval by ID and date-based listing.

## Security

Secrets are loaded from environment variables.

The following should never be committed:

* `.env`
* clinical audio files
* clinical transcript files
* API keys
* database passwords