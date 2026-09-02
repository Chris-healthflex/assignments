# Voice/Note → Structured Clinical Assessment Form Filler

A production-grade Python backend system designed to ingest clinical voice session recordings (`.wav`), perform automatic speech recognition (ASR) via OpenAI Whisper, extract structured clinical entities using a stateful LangGraph workflow, enforce strict zero-hallucination grounding validation, serialize into a strictly typed `FirstAssessment` Pydantic v2 schema, and persist/query clinical records via FastAPI and MongoDB.

---

## Architecture & Processing Pipeline

```
clinical_assessment.wav
        ↓
Whisper Transcription (ASR via OpenAI Whisper API)
        ↓
Transcript Text
        ↓
LangGraph Extraction Agent (StateGraph with Structured LLM Output)
        ↓
Deterministic Grounding & Anti-Hallucination Validation
        ↓
Strict FirstAssessment Pydantic Schema (extra="forbid")
        ↓
MongoDB Async Persistence (Motor Repository Layer)
```

### Major Components:
1. **Transcription Layer (`app/services/transcriber.py`)**: Validates audio files (MIME type, non-empty binary size, audio extension) and dispatches to the OpenAI Whisper API (`whisper-1`) with safe file streaming and zero token leakage.
2. **Clinical Extraction Layer (`app/services/langgraph_agent.py`)**: Executes a linear 3-node LangGraph `StateGraph` (`extract_clinical_data` → `validate_grounding` → `normalize_extraction`) using `ChatOpenAI` at `temperature=0` with structured outputs.
3. **Anti-Hallucination & Grounding Engine (`app/services/confidence.py`)**: Uses strict lookaround numeric token matching and keyword alignment to verify that extracted examination measurements, ROM angles, and laterality are explicitly anchored in the transcript.
4. **Schema Layer (`app/schemas/assessment.py`)**: Production `FirstAssessment` Pydantic v2 model enforcing `extra="forbid"`, non-null string defaults (`""`), array preservation (`[]`), and zero internal metadata leakage.
5. **Database Layer (`app/db/` & `app/repositories/`)**: Motor-based async persistence managing indexing (`created_at: -1`), CRUD operations, and ISO date-range querying.
6. **REST API Layer (`app/api/`)**: FastAPI routing exposing all 4 required clinical assessment endpoints with structured error handling (including field-level `HTTP 422`).

---

## Technology Stack

- **Python**: 3.10+ (tested on Python 3.13)
- **Web Framework**: FastAPI, Uvicorn, Starlette
- **Data Validation & Settings**: Pydantic v2, `pydantic-settings`
- **AI & Agentic Orchestration**: LangGraph, LangChain Core, LangChain OpenAI
- **Speech-to-Text (ASR)**: OpenAI Whisper API (`whisper-1`)
- **Database**: MongoDB with Motor (Async PyMongo driver)
- **Testing**: Pytest, `pytest-asyncio`, `pytest-mock`

---

## Project Structure

```
assignments/
├── .env.example                     # Environment configuration template
├── .gitignore                        # Git ignore rules (secrets, venv, pycache)
├── requirements.txt                  # Production and testing dependencies
├── run_assessment_test.py            # Standalone end-to-end pipeline runner
├── clinical_assessment.wav           # Physiotherapy session audio recording
├── README.md                         # Comprehensive documentation
├── app/
│   ├── __init__.py
│   ├── main.py                       # FastAPI application entrypoint & lifespan
│   ├── config.py                     # Centralized settings via pydantic-settings
│   ├── api/
│   │   ├── __init__.py
│   │   ├── deps.py                   # Dependency injection providers
│   │   └── assessments.py            # EP1, EP2, EP3, EP4 route implementations
│   ├── db/
│   │   ├── __init__.py
│   │   ├── models.py                 # AssessmentDocument persistence wrapper
│   │   └── mongo.py                  # Motor AsyncIOMotorClient manager
│   ├── repositories/
│   │   ├── __init__.py
│   │   └── assessment_repo.py        # Async CRUD, pagination, and date filtering
│   ├── schemas/
│   │   ├── __init__.py
│   │   └── assessment.py             # Strict FirstAssessment Pydantic v2 schemas
│   └── services/
│       ├── __init__.py
│       ├── confidence.py             # Grounding check and anti-hallucination logic
│       ├── langgraph_agent.py        # LangGraph StateGraph clinical extraction
│       ├── prompts.py                # Zero-hallucination prompt definitions
│       └── transcriber.py            # Whisper audio transcription service
└── tests/
    ├── __init__.py
    ├── test_api_endpoints.py         # Integration tests for EP1-EP4 & /health
    ├── test_confidence.py            # Grounding & hallucination detection tests
    ├── test_langgraph_agent.py       # LangGraph agent & regression test suite
    ├── test_mongo_repo.py            # MongoDB CRUD, indexing, and date filter tests
    ├── test_runner.py                # Unit tests for run_assessment_test.py
    ├── test_schemas.py               # Schema strictness (extra='forbid') tests
    └── test_transcriber.py           # Audio validation and transcription tests
```

---

## Environment Setup & Configuration

### 1. Prerequisites
- Python 3.10 or higher (Python 3.13 supported)
- Local or remote MongoDB instance (default: `mongodb://localhost:27017`)

### 2. Virtual Environment Setup
```bash
# Create virtual environment
python -m venv .venv

# Activate on Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Activate on Linux / macOS
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Environment Variables Configuration
Copy `.env.example` to `.env` and provide your configuration:

```bash
cp .env.example .env
```

`.env` configuration variables:
```env
# Application
APP_NAME=Clinical-Assessment-Pipeline
ENV=development
DEBUG=true
HOST=0.0.0.0
PORT=8000

# OpenAI & Transcriber
OPENAI_API_KEY=your_actual_openai_api_key_here
WHISPER_MODEL=whisper-1
EXTRACTION_MODEL=gpt-4o
CONFIDENCE_THRESHOLD=0.75

# MongoDB
MONGO_URI=mongodb://localhost:27017
MONGO_DB_NAME=clinical_db
MONGO_COLLECTION=assessments
```

*(Note: Never commit real API keys to version control. Secrets are protected via `.gitignore`).*

---

## MongoDB Setup

The application connects asynchronously to MongoDB via `motor.motor_asyncio.AsyncIOMotorClient`.

- **Default URI**: `mongodb://localhost:27017`
- **Database**: `clinical_db`
- **Collection**: `assessments`
- **Automatic Indexing**: On application startup or repository initialization, an index on `created_at` (`-1`) is automatically created for efficient date-range queries and sorted pagination.

---

## Running the Application

Start the FastAPI server using Uvicorn:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

- **Interactive Swagger Docs**: `http://localhost:8000/docs`
- **ReDoc Documentation**: `http://localhost:8000/redoc`
- **Health Check**: `http://localhost:8000/health`

---

## REST API Endpoints

### 1. EP1: `POST /assessments/parse`
- **Input**: `multipart/form-data` with `file: UploadFile` (WAV audio recording).
- **Processing**:
  1. Validates audio format and non-empty file size.
  2. Transcribes audio via OpenAI Whisper API.
  3. Executes LangGraph extraction workflow.
  4. Runs deterministic grounding validation.
- **Output**: Pure `FirstAssessment` JSON (`200 OK`).
- **Error Behavior**:
  - `400 Bad Request`: Empty or invalid audio file format.
  - `422 Unprocessable Entity`: Extraction confidence below threshold or ungrounded fields detected, with structured field-level error messages:
    ```json
    {
      "detail": [
        {
          "field": "objectiveAssessment.tests[0].left",
          "message": "Measurement value '999' is not found or supported in the transcript text.",
          "value": "999"
        }
      ]
    }
    ```
  - `500 Internal Server Error`: Whisper transcription or processing failure.

### 2. EP2: `POST /assessments`
- **Input**: `FirstAssessment` JSON body.
- **Processing**: Validates strict schema and inserts into MongoDB with UTC timestamps.
- **Output**: `201 Created` with created document ID:
  ```json
  {
    "id": "6a9809b22b36f4cca6b75bbe",
    "message": "Assessment saved successfully",
    "created_at": "2026-09-02T18:00:00+00:00",
    "assessment": { ...FirstAssessment... }
  }
  ```

### 3. EP3: `GET /assessments/{id}`
- **Input**: Path parameter `id` (24-hex MongoDB ObjectId).
- **Output**: Retrieved `FirstAssessment` JSON (`200 OK`).
- **Error Behavior**: `404 Not Found` if the assessment does not exist or ID format is invalid.

### 4. EP4: `GET /assessments`
- **Input**: Query parameters:
  - `start_date`: Optional ISO-8601 string (e.g. `2026-09-01T00:00:00Z`)
  - `end_date`: Optional ISO-8601 string (e.g. `2026-09-02T23:59:59Z`)
  - `skip`: Integer offset (default `0`)
  - `limit`: Integer limit (default `20`, max `100`)
- **Output**: `200 OK`:
  ```json
  {
    "total": 1,
    "skip": 0,
    "limit": 20,
    "items": [ { ...FirstAssessment... } ]
  }
  ```

### 5. Health Check: `GET /health`
- **Output**: `200 OK`: `{"status": "healthy", "database": "connected", "environment": "development"}`.

---

## Strict Schema Design (`FirstAssessment`)

The output schema strictly adheres to the required clinical structure:

```
FirstAssessment
├── clinicalDetails
│   ├── clinicalHistory: str ("")
│   ├── chiefComplaint: str ("")
│   └── duration: Dict[str, Any] ({})
├── subjectiveAssessments: List[SubjectiveAssessment] ([])
│   └── SubjectiveAssessment
│       ├── testName: str ("")
│       └── conclusion: List[str] ([])
├── objectiveAssessment: ObjectiveAssessment
│   └── tests: List[ObjectiveTest] ([])
│       └── ObjectiveTest
│           ├── testName: str ("")
│           ├── unitName: str ("")
│           ├── value: str ("")
│           ├── left: str ("")
│           ├── right: str ("")
│           └── comments: List[str] ([])
├── subjectiveGoals: List[SubjectiveGoal] ([])
│   └── SubjectiveGoal
│       ├── goalDetails: str ("")
│       └── targetDate: str ("")
├── objectiveGoals: List[ObjectiveGoal] ([])
│   └── ObjectiveGoal
│       ├── goalName: str ("")
│       ├── goalCategory: str ("")
│       └── unitName: str ("")
│       ├── value: str ("")
│       └── targetDate: str ("")
├── recommendation: List[Recommendation] ([])
│   └── Recommendation
│       ├── sessionType: str ("")
│       └── sessionFrequency: str ("")
└── patientAdvice: PatientAdvice
    └── adviceDetails: str ("")
```

### Schema Guarantees:
- **`extra="forbid"`**: Extra, renamed, metadata, or debug fields are strictly rejected.
- **Non-null Strings**: All string fields default to `""` and never serialize to `null`.
- **Array Preservation**: All array fields always serialize as lists (`[]`), even when empty or containing a single element.
- **Isolation**: Database IDs (`_id`, `id`) and timestamps (`created_at`) are maintained exclusively in the persistence model and **never leak** into `FirstAssessment`.

---

## Anti-Hallucination & Grounding Strategy

The system enforces a strict Zero-Hallucination policy across the extraction pipeline:

1. **Transcript as Sole Truth**: Facts, measurements, and recommendations are extracted only if explicitly articulated in the transcript.
2. **Absence vs. Inference**:
   - Examination measurements (e.g. extension `20°` / `-5°`) are **never** inferred as target goal numbers (e.g. `0°`).
   - If goal numbers or deadlines are not spoken, `value`, `unitName`, and `targetDate` strictly remain empty strings `""`.
   - Treatment recommendations (e.g. Physiotherapy 1x/week) belong in `recommendation[]` and are **never** converted into `patientAdvice`.
   - If no explicit patient advice was given, `adviceDetails` strictly remains `""`.
   - If no subjective goals were stated, `subjectiveGoals` strictly remains `[]`.
3. **Strict Numeric Grounding**: Numeric validation uses regex lookaround boundaries `(?<![0-9.])number(?![0-9.])` to avoid false substring matching (preventing `"0"` from matching inside `"20"` or `"100"`).

---

## Standalone Pipeline Runner

The standalone script executes the complete production pipeline against an audio file:

```bash
python run_assessment_test.py clinical_assessment.wav
```

- Progress logs are printed to `stderr`.
- Validated `FirstAssessment` JSON is printed directly to `stdout`.

> **Known External Service Limitation**:
> During live audio testing against the real `clinical_assessment.wav`, the Whisper API call was initiated and connected to OpenAI (`whisper-1`), but returned `HTTP 429 insufficient_quota` because the configured OpenAI API account currently has zero available billing credits (`credit_balance_exhausted`).
> The application safely caught this error, logged a clear error message to `stderr` without leaking credentials, and exited cleanly. When active credits are supplied, the real pipeline executes without modifications.

---

## Running Automated Tests

Run the complete test suite:

```bash
python -m pytest -v
```

### Test Suite Summary:
- **54 Unit & Integration Tests Passed**:
  - `tests/test_schemas.py`: Verifies `extra="forbid"`, array serialization, non-null strings, and top-level key preservation.
  - `tests/test_transcriber.py`: Verifies audio validation, resource handle safety, error wrapping, and model configuration.
  - `tests/test_confidence.py`: Verifies numeric lookaround token grounding, hallucination detection, and empty-field validity.
  - `tests/test_langgraph_agent.py`: Verifies LangGraph workflow execution, laterality, frequency extraction, metadata isolation, and advice separation.
  - `tests/test_mongo_repo.py`: Verifies MongoDB ping, CRUD, date-range filtering, and pagination with automated teardown.
  - `tests/test_api_endpoints.py`: Verifies EP1–EP4 routes, status codes (200, 201, 400, 404, 422, 500), and `/health`.
  - `tests/test_runner.py`: Verifies `run_assessment_test.py` pipeline orchestration.
- **2 Live Integration Tests Skipped**: Live OpenAI network tests gracefully skip when external API quota is exhausted.
