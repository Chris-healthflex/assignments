# Clinical Assessment Pipeline

An end-to-end clinical audio assessment service for the Stance Health assignment. The service accepts a clinician-patient WAV recording, transcribes it locally with Whisper, extracts structured assessment data with a LangGraph workflow and Groq, validates the result against the `FirstAssessment` schema, and stores submitted assessments in MongoDB.

The project intentionally contains no frontend. It exposes a FastAPI service and interactive Swagger documentation.

## Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Technology Stack](#technology-stack)
- [Assessment Schema](#assessment-schema)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the Service](#running-the-service)
- [API Usage](#api-usage)
- [Running the Pipeline Script](#running-the-pipeline-script)
- [Testing](#testing)
- [Design Decisions](#design-decisions)
- [Failure Behavior](#failure-behavior)
- [Scope and Limitations](#scope-and-limitations)

## Overview

The service implements the required audio-to-assessment workflow:

1. Accept a WAV upload at `POST /assessments/parse`.
2. Transcribe the audio with the local Whisper `base` model.
3. Send the transcript to a LangGraph extraction workflow using Groq model `openai/gpt-oss-120b`.
4. Produce a strict Pydantic `FirstAssessment` object.
5. Reject low-confidence extraction results with field-level HTTP 422 details.
6. Return the validated JSON response.
7. Persist an assessment through `POST /assessments`.
8. Retrieve one assessment or list assessments from MongoDB.

The parser endpoint and persistence endpoint are deliberately separate. `POST /assessments/parse` extracts an assessment but does not save it. `POST /assessments` saves an already parsed and validated assessment and returns the generated MongoDB ID.

## Architecture

```text
                         POST /assessments/parse
                                  |
                                  v
                           WAV validation
                                  |
                                  v
                       Temporary audio file
                                  |
                                  v
                        Local Whisper base
                                  |
                                  v
                             Transcript
                                  |
                                  v
                    LangGraph + Groq extraction
                                  |
                                  v
               FirstAssessment + field confidence
                                  |
                 +----------------+----------------+
                 |                                 |
        confidence below 0.75              confidence accepted
                 |                                 |
                 v                                 v
        HTTP 422 field details              FirstAssessment JSON
                                                   |
                                                   v
                              POST /assessments -> MongoDB Atlas
                                                   |
                              +--------------------+-------------------+
                              |                                        |
                              v                                        v
                  GET /assessments/{id}                 GET /assessments
```

### Component responsibilities

- `app/main.py`: creates the FastAPI application and registers routes.
- `app/api/assessments.py`: handles upload parsing, endpoint validation, confidence gating, and route responses.
- `app/pipeline/transcription.py`: validates WAV files and calls local Whisper.
- `app/pipeline/extraction.py`: defines the extraction state, strict model output wrapper, system prompt, and LangGraph workflow.
- `app/pipeline/mapping.py`: maps and validates extraction results and identifies low-confidence fields.
- `app/models/first_assessment.py`: defines the exact assessment schema.
- `app/models/api_models.py`: defines persisted API records with `id` and `created_at` metadata.
- `app/db/connection.py`: creates the reusable MongoDB client and assessment collection.
- `app/db/assessments.py`: saves, retrieves, and lists assessments.
- `tests/run_pipeline.py`: runs the complete WAV-to-JSON workflow from the command line.

## Project Structure

```text
assignments/
|-- app/
|   |-- api/
|   |   |-- assessments.py
|   |-- core/
|   |   |-- config.py
|   |-- db/
|   |   |-- assessments.py
|   |   |-- connection.py
|   |-- models/
|   |   |-- api_models.py
|   |   |-- first_assessment.py
|   |-- pipeline/
|       |-- extraction.py
|       |-- mapping.py
|       |-- transcription.py
|   |-- main.py
|-- tests/
|   |-- fixtures/
|   |-- run_pipeline.py
|   |-- test_api.py
|   |-- test_multi_array_regression.py
|   |-- test_no_hallucination.py
|   |-- test_pipeline.py
|   |-- test_schema.py
|-- clinical_assessment.wav       # Local assignment input, if supplied
|-- .env.example
|-- .gitignore
|-- pyproject.toml
|-- README.md
```

## Technology Stack

- Python 3.10+
- FastAPI and Uvicorn
- Pydantic v2 and pydantic-settings
- OpenAI Whisper, running locally with the `base` model
- LangGraph for the extraction workflow
- LangChain Groq integration
- Groq `openai/gpt-oss-120b` for structured extraction
- PyMongo for MongoDB Atlas persistence
- Pytest and `mongomock` for deterministic tests
- Git for version control and candidate-branch submission

The extraction provider originally targeted OpenAI `gpt-4o-mini`, but the OpenAI path was blocked by quota. The implementation migrated to Groq `openai/gpt-oss-120b`; the configured model can be changed through `EXTRACTION_MODEL`.

## Assessment Schema

The output follows the supplied `FirstAssessment` production schema:

```json
{
  "clinicalDetails": {
    "clinicalHistory": "",
    "chiefComplaint": "",
    "duration": ""
  },
  "subjectiveAssessments": [
    {
      "testName": "",
      "conclusion": ""
    }
  ],
  "objectiveAssessment": {
    "tests": [
      {
        "testName": "",
        "unitName": "",
        "value": "",
        "left": "",
        "right": "",
        "comments": ""
      }
    ]
  },
  "subjectiveGoals": [
    {
      "goalDetails": "",
      "targetDate": ""
    }
  ],
  "objectiveGoals": [
    {
      "goalName": "",
      "goalCategory": "",
      "unitName": "",
      "value": "",
      "targetDate": ""
    }
  ],
  "recommendation": [
    {
      "sessionType": "",
      "sessionFrequency": ""
    }
  ],
  "patientAdvice": {
    "adviceDetails": ""
  }
}
```

Schema rules:

- Top-level keys and nested key names are case-sensitive.
- Array fields remain arrays, including when there is only one item or no items.
- All visible leaf fields are strings.
- Missing string values are represented by `""`, never `null`.
- Missing repeated sections are represented by `[]`.
- Unknown fields are rejected by Pydantic.
- The extraction prompt prohibits invented clinical values, scores, dates, diagnoses, measurements, and recommendations.

## Installation

From the `assignments` directory:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[test]"
Copy-Item .env.example .env
```

The editable install includes the application dependencies and test dependencies. Whisper may require `ffmpeg` to be installed and available on `PATH`.

## Configuration

Copy `.env.example` to `.env` and set the values for your environment:

```dotenv
MONGODB_URI=mongodb+srv://<username>:<password>@<cluster-url>/?retryWrites=true&w=majority
MONGODB_DATABASE=clinical_assessment
MONGODB_COLLECTION=assessments
WHISPER_MODEL=base
GROQ_API_KEY=<your-groq-api-key>
EXTRACTION_MODEL=openai/gpt-oss-120b
CONFIDENCE_THRESHOLD=0.75
```

### MongoDB Atlas

For Atlas:

1. Create a database user.
2. Add the development machine IP address to the Atlas network access list.
3. Copy the Atlas connection string into `MONGODB_URI`.
4. Set `MONGODB_DATABASE` and `MONGODB_COLLECTION`.
5. Keep `.env` untracked. It is covered by `.gitignore`.

For local MongoDB instead:

```powershell
docker run -d -p 27017:27017 --name clinical-mongo mongo:7
```

The MongoDB client is created lazily and reused through a cached connection. Assessment documents are stored with the validated `FirstAssessment` fields plus internal `created_at` metadata. MongoDB's `_id` is returned through the API as the string field `id`.

## Running the Service

Start the development server:

```powershell
uvicorn app.main:app --reload
```

Open the interactive API documentation:

```text
http://127.0.0.1:8000/docs
```

The ReDoc view is available at:

```text
http://127.0.0.1:8000/redoc
```

## API Usage

### 1. Parse a WAV file

`POST /assessments/parse` accepts a multipart upload named `file` and returns a validated `FirstAssessment`. It does not save the result.

```powershell
curl.exe -X POST "http://127.0.0.1:8000/assessments/parse" `
  -H "accept: application/json" `
  -F "file=@clinical_assessment.wav;type=audio/wav"
```

Successful response: HTTP 200 with the exact `FirstAssessment` JSON shape.

### 2. Save an assessment

`POST /assessments` accepts a `FirstAssessment` JSON body and saves it to MongoDB.

```powershell
curl.exe -X POST "http://127.0.0.1:8000/assessments" `
  -H "accept: application/json" `
  -H "Content-Type: application/json" `
  --data-binary "@assessment.json"
```

The response is HTTP 201 and includes the generated MongoDB ID:

```json
{
  "id": "6a87ef48af746e60a086e529",
  "created_at": "2026-08-21T06:25:12.705115",
  "clinicalDetails": {},
  "subjectiveAssessments": [],
  "objectiveAssessment": {"tests": []},
  "subjectiveGoals": [],
  "objectiveGoals": [],
  "recommendation": [],
  "patientAdvice": {}
}
```

Use the returned `id` with the retrieval endpoint.

### 3. Retrieve one assessment

```powershell
curl.exe "http://127.0.0.1:8000/assessments/6a87ef48af746e60a086e529"
```

Returns HTTP 200 when the ID exists, or HTTP 404 when it does not.

### 4. List assessments

List all saved assessments:

```powershell
curl.exe "http://127.0.0.1:8000/assessments"
```

Filter by ISO-8601 creation dates:

```powershell
curl.exe "http://127.0.0.1:8000/assessments?from_date=2026-08-01T00:00:00&to_date=2026-08-31T23:59:59"
```

Malformed date values return HTTP 400.

## Running the Pipeline Script

Run the complete audio-to-JSON pipeline on the supplied WAV:

```powershell
python tests/run_pipeline.py clinical_assessment.wav
```

The script:

1. Validates the input file.
2. Runs local Whisper transcription.
3. Runs LangGraph and Groq extraction.
4. Applies the confidence threshold.
5. Prints the validated `FirstAssessment` JSON.

This script prints the parsed JSON but does not save it to MongoDB. Use the returned JSON as the body of `POST /assessments` to persist it.

## Testing

Run the full suite:

```powershell
python -m pytest -v
```

The tests are designed to pass without a live Groq API, Whisper model download, or MongoDB connection:

- `test_schema.py`: verifies exact top-level and nested schema keys, array types, string defaults, and rejection of unknown fields.
- `test_no_hallucination.py`: verifies that absent dates, measurements, goals, and diagnoses remain empty rather than being fabricated.
- `test_pipeline.py`: verifies WAV validation and confidence filtering.
- `test_multi_array_regression.py`: verifies preservation of five goals and five tests.
- `test_api.py`: verifies success and failure behavior for all four endpoints and tests MongoDB persistence with `mongomock`.

The real pipeline script requires a configured `GROQ_API_KEY`, the local Whisper model, and the supplied WAV file. Live Atlas verification requires a valid `MONGODB_URI` and network access.

## Design Decisions

### Strict schema boundary

Pydantic v2 models are the final output boundary. Models use `extra="forbid"`, so accidental fields cannot silently reach the frontend. The LangGraph structured-output wrapper also uses the same assessment model, reducing the chance of a model response drifting from the API contract.

### No hallucination policy

The extraction prompt explicitly instructs the model to use empty strings or arrays when the transcript does not provide a value. It must not invent clinical measurements, scores, dates, diagnoses, or recommendations. Tests mock the model response to make these guarantees deterministic.

### Confidence gate

The extraction output includes a confidence map keyed by field or dotted field path. The configured threshold is `0.75`. Any reported field below the threshold causes HTTP 422 with the field name, score, and reason so a caller can send the assessment for human review instead of silently accepting uncertain data.

### Local Whisper

Whisper runs locally with the `base` model. This avoids introducing a second hosted transcription API and keeps the audio-to-text stage under the application's control. The tradeoff is local model download time and CPU cost.

### Groq extraction provider

The extraction provider is isolated inside `ClinicalExtractionGraph`. This keeps the application route independent of provider-specific details and allows the configured model to change through `EXTRACTION_MODEL`. Groq is currently used because the original OpenAI target was blocked by quota.

### Temporary audio handling

Uploaded audio is written to a temporary file for Whisper and deleted in a `finally` block after processing. The API does not persist raw audio in MongoDB.

### Persistence model

The validated assessment payload is stored as-is. MongoDB metadata is kept outside the `FirstAssessment` schema: `_id` is converted to the response field `id`, and `created_at` is returned only by persistence endpoints.

### No frontend

The assignment requires a backend pipeline and explicitly excludes UI work. Swagger and ReDoc are provided by FastAPI for API inspection, but no custom frontend, static assets, or HTML application is included.

## Failure Behavior

- Non-WAV or empty uploads: HTTP 422.
- Missing Whisper dependency or empty transcript: controlled transcription error.
- Empty transcript: extraction error.
- Low-confidence extraction: HTTP 422 with field-level details.
- Invalid `FirstAssessment` JSON or unknown fields: FastAPI validation HTTP 422.
- Invalid assessment ID or missing record: HTTP 404.
- Malformed date filters: HTTP 400.
- MongoDB connection or write failures: surfaced as server errors and should be monitored by the deployment environment.

## Scope and Limitations

Included:

- The required WAV-to-JSON clinical assessment pipeline.
- The four required assessment endpoints.
- MongoDB persistence and date filtering.
- Strict schema validation and hallucination protection.
- Automated tests and evaluator documentation.

Not included:

- A custom frontend or clinician dashboard.
- Authentication and authorization.
- Clinical diagnosis beyond faithfully extracting transcript content.
- Background job processing or a queue.
- Deployment infrastructure.
- Persisting raw audio files.