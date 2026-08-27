# Clinical Audio to Structured Assessment Report Pipeline

A production-grade Python/FastAPI pipeline that ingests clinical audio session recordings (WAV), transcribes them using **OpenAI Whisper**, extracts structured clinical data using **LangGraph / LangChain** with strict anti-hallucination guarantees, validates against the strict production **FirstAssessment Pydantic v2 schema**, and persists the structured assessment into **MongoDB**.

---

## 1. Overview & Architecture

```
[ Clinical Audio (.wav) ]
           │
           ▼
[ Whisper Transcription Module ]  ── (OpenAI Whisper / Local Whisper / API)
           │ (Transcribed Dialogue)
           ▼
[ LangGraph Extraction Agent ]    ── (Anti-Hallucination & Factual Grounding Guard)
           │
           ▼
[ Pydantic v2 Schema Enforcement ]── (Exact 7-Section FirstAssessment Schema)
           │
           ├─ [Confidence < Threshold] ──► HTTP 422 Unprocessable Entity
           │
           ▼ [Validated JSON]
[ FastAPI REST Endpoints ]        ──► [ MongoDB Persistent Storage (PyMongo/Motor) ]
```

---

## 2. Exact Output Schema (7 Sections)

The pipeline produces JSON conforming strictly to the frontend production schema:

1. **`clinicalDetails`**:
   - `clinicalHistory`: string
   - `chiefComplaint`: string
   - `duration`: string
2. **`subjectiveAssessments`**: Array of objects:
   - `testName`: string
   - `conclusion`: string
3. **`objectiveAssessment`**:
   - `tests`: Array of objects:
     - `testName`: string
     - `unitName`: string
     - `value`: string
     - `left`: string
     - `right`: string
     - `comments`: string
4. **`subjectiveGoals`**: Array of objects:
   - `goalDetails`: string
   - `targetDate`: string
5. **`objectiveGoals`**: Array of objects:
   - `goalName`: string
   - `goalCategory`: string
   - `unitName`: string
   - `value`: string
   - `targetDate`: string
6. **`recommendation`**: Array of objects:
   - `sessionType`: string
   - `sessionFrequency`: string
7. **`patientAdvice`**:
   - `adviceDetails`: string

### Strict Integrity Rules
- **No Hallucination**: Medical details not present in the audio are never fabricated.
- **No Extra Keys**: `extra = "forbid"` prevents unexpected keys.
- **No Nulls**: All string fields are guaranteed to be strings (`""` if absent).
- **Arrays**: All list fields are guaranteed to be JSON arrays (`[]`), even for single or empty items.

---

## 3. API Endpoints

### **EP1 — Parse Audio Session**
- **Method / Path**: `POST /assessments/parse`
- **Request**: Multipart Form (`file: clinical_assessment.wav`)
- **Success (200 OK)**: Returns structured `FirstAssessment` JSON.
- **Low Confidence / Invalid Audio (422 Unprocessable Entity)**: Returns field-level errors and confidence metrics.
- **Invalid Format (400 Bad Request)**: When uploaded file is not a valid WAV file.

### **EP2 — Save Assessment**
- **Method / Path**: `POST /assessments`
- **Request Body**: `FirstAssessment` JSON object.
- **Response (201 Created)**: Returns `{ "id": "<mongo_id>", "assessment": {...}, "created_at": "<iso_timestamp>" }`.

### **EP3 — Retrieve Assessment by ID**
- **Method / Path**: `GET /assessments/{id}`
- **Response (200 OK)**: Returns saved assessment record.
- **Not Found (404)**: When ID does not exist in MongoDB.

### **EP4 — List Assessments**
- **Method / Path**: `GET /assessments`
- **Query Parameters**:
  - `date`: Filter by exact date prefix (e.g. `?date=2026-08-27`)
  - `start_date` / `end_date`: Filter by ISO date-time range
  - `skip` / `limit`: Pagination controls
- **Response (200 OK)**: Returns `{ "total": <int>, "items": [...] }`.

---

## 4. Environment Variables & Configuration

Create a `.env` file in the root directory (refer to `.env.example`):

```env
# Application
APP_NAME=Clinical Audio Assessment Pipeline
DEBUG=False

# MongoDB Database Configuration
MONGODB_URI=mongodb://localhost:27017
MONGODB_DB_NAME=clinical_assessments_db
MONGODB_COLLECTION=assessments

# OpenAI Configuration (Optional, for OpenAI API extraction & Whisper API)
OPENAI_API_KEY=your_openai_api_key_here
LLM_MODEL=gpt-4o-mini
LLM_TEMPERATURE=0.0

# Whisper Configuration (options: "local", "api", "mock")
WHISPER_MODE=local
WHISPER_MODEL_SIZE=base

# Extraction Confidence Settings
MIN_CONFIDENCE_THRESHOLD=0.50
```

---

## 5. Setup & Installation

### Prerequisites
- Python 3.10+ (tested with Python 3.11 / 3.12)
- MongoDB instance running locally on `localhost:27017` (or MongoDB Atlas connection string in `.env`). *Note: In-memory fallback is automatically engaged if MongoDB is not reachable.*

### Installation Steps

1. **Clone repository and navigate into project directory**:
   ```bash
   git clone <repo_url>
   cd myown
   ```

2. **Create and activate virtual environment**:
   ```bash
   # On Windows PowerShell
   py -3.11 -m venv .venv
   .venv\Scripts\activate

   # On Linux/macOS
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 6. How to Run

### Option A: Run the End-to-End Pipeline Script (D5)
Executes the full pipeline (WAV generation/loading → Whisper → LangGraph Extraction → Validation → MongoDB Save → Retrieve & Print JSON):

```bash
python run_pipeline.py --audio clinical_assessment.wav
```

### Option B: Run the FastAPI Server
Start the development server with live reload:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Access Interactive API Documentation:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

---

## 7. Running Automated Tests

Run the complete test suite with `pytest`:

```bash
pytest tests/ -v
```

Tests include:
- `test_schema.py`: Pydantic v2 strict models, no nulls, extra-field forbidding.
- `test_transcription.py`: WAV header verification and Whisper execution.
- `test_extraction.py`: LangGraph extraction agent, anti-hallucination validation, low-confidence flagging.
- `test_api.py`: All 4 REST endpoints (EP1–EP4) including HTTP 200, 201, 400, 404, 422.

---

## 8. Design Decisions, Assumptions & Limitations

### Design Decisions
1. **Pydantic v2 Strict Mode (`extra = "forbid"`)**: Guarantees zero schema drift and rejects unknown fields immediately.
2. **Deterministic Anti-Hallucination Guard**: A dedicated state-graph node cross-checks all extracted tokens against the transcribed dialogue. Any value not grounded in dialogue is discarded, preventing dangerous medical hallucinations.
3. **Resilient MongoDB Layer**: PyMongo integration includes automated fallback to an in-memory document storage engine when running in environments without a running MongoDB daemon (e.g. CI/CD or lightweight local testing).
4. **Dual Whisper Backends**: Supports local Whisper models (offline zero-cost inference) and OpenAI Whisper API (`whisper-1`) seamlessly via configuration.

### Assumptions & Limitations
- Audio inputs are assumed to be 16-bit uncompressed PCM WAV files.
- If an audio recording is corrupted or contains inaudible noise with confidence below `0.50`, the service safely rejects extraction with HTTP 422 to prevent erroneous clinical reports.
