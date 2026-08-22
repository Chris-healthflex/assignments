# Clinical Assessment Voice Form Filler

A complete, production-ready FastAPI service that automates the extraction of structured clinical assessments from WAV audio recordings. It transcribes audio files, executes a 5-node LangGraph extraction and normalization workflow, validates confidence levels, and persists data to MongoDB Atlas.

---

## Architecture and Flow

```
[Audio Upload] ──> [Whisper Service (Groq/Local)] ──> [Transcript]
                                                           │
┌────────────────────────── LangGraph Pipeline ────────────┘
│
└─> START ──> [extract_node] ──> [normalize_node] ──> [confidence_node] ──> [validate_node] ──> END
                    │                   │                     │                    │
                    ▼                   ▼                     ▼                    ▼
             JSON Extraction     Format Dates/Units     LLM Audit against     Confidence
             w/ Pydantic validation  (Qwen/Llama)        raw transcript        check vs 0.70
```

---

## Features

- **Double-Safe Audio Validation:** Uploads are checked for both extension (`.wav`) and binary headers (`RIFF` and `WAVE` file signatures) to prevent malicious or malformed uploads.
- **Robust Transcription (Whisper):** Selected dynamically via `WHISPER_PROVIDER=groq` (remotely hosted `whisper-large-v3-turbo`) or `WHISPER_PROVIDER=local` (offline fallback using `faster-whisper` on CPU).
- **LangGraph 5-Node Workflow:** Standardized, deterministic graph execution flow for data processing.
- **Self-Healing LLM Extraction:** Re-tries failed Pydantic validation attempts (up to 3 times), feeding validation errors back to the model before falling back to default values.
- **Automated Rate-Limit (429) Handling:** Built-in exponential backoff and retry wrapper to safely execute under free-tier token limits (TPM).
- **Strict Schema Invariants:** No null values allowed (strings default to `""`, arrays to `[]`), uses `extra="forbid"`, and preserves casing and values exactly as spoken.
- **Granular Confidence Checks:** Evaluates populated fields against raw transcript text using an LLM audit. Fields below `CONFIDENCE_THRESHOLD` (default `0.70`) trigger a `422 Unprocessable Entity` response containing field-level reasons and scores, preventing dirty data writes.
- **MongoDB Atlas Integration:** Thread-safe, singleton client instantiation (using Lifespan events) with database metadata (`created_at`, `updated_at` indexes) kept separate from public response envelopes.

---

## Environment Variables (`.env`)

Create a `.env` file at the root of the workspace (already registered in `.gitignore`).

```ini
# MongoDB Atlas Configuration
MONGODB_URI=mongodb+srv://<user>:<password>@cluster.mongodb.net/?appName=clinical-assessment
MONGODB_DATABASE=clinical_assessment

# LLM Extraction Settings (Groq API or Local Ollama)
LLM_PROVIDER=groq
LLM_MODEL=groq/compound-mini
GROQ_API_KEY=gsk_your_key_here
GROQ_BASE_URL=https://api.groq.com
OLLAMA_BASE_URL=http://localhost:11434

# Audio Transcription Settings
WHISPER_PROVIDER=groq
WHISPER_MODEL=whisper-large-v3-turbo
WHISPER_DEVICE=cpu
WHISPER_COMPUTE_TYPE=int8

# Confidence Auditing Configuration
CONFIDENCE_THRESHOLD=0.70
```

---

## Installation & Setup

1. **Create and Activate Virtual Environment:**
   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   ```

2. **Install Dependencies:**
   ```powershell
   python -m pip install -r requirements.txt
   ```
   *(For offline fallback, install optional packages: `pip install faster-whisper`)*

3. **Start FastAPI Application:**
   ```powershell
   .venv\Scripts\uvicorn app.main:app --reload --port 8000
   ```
   Open Swagger UI at `http://127.0.0.1:8000/docs`.

---

## REST API Endpoints

### 1. `POST /assessments/parse`
- **Description:** Accept WAV recording, transcribe, extract fields, validate confidence, persist to MongoDB Atlas.
- **Payload:** Multipart form-data with key `file`.
- **Response (201 Created):**
  ```json
  {
    "id": "66c72b2f9b1d8b2d88888888",
    "assessment": {
      "clinicalDetails": {
        "clinicalHistory": "Left tibial condyle fracture...",
        "chiefComplaint": "Left knee pain...",
        "duration": "8 months"
      },
      "subjectiveAssessments": [],
      "objectiveAssessment": {
        "tests": [
          {
            "testName": "Knee flexion",
            "unitName": "degrees",
            "value": "",
            "left": "124",
            "right": "130",
            "comments": ""
          }
        ]
      },
      "subjectiveGoals": [],
      "objectiveGoals": [],
      "recommendation": [
        {
          "sessionType": "Physiotherapy",
          "sessionFrequency": "once weekly for 4 sessions"
        }
      ],
      "patientAdvice": {
        "adviceDetails": ""
      }
    }
  }
  ```
- **Error Response (422 Unprocessable Entity):**
  If any field falls below the confidence threshold:
  ```json
  {
    "detail": [
      {
        "field": "clinicalDetails.duration",
        "reason": "Transcript mentions knee pain but does not state 8 months",
        "confidence": 0.35
      }
    ]
  }
  ```

### 2. `GET /assessments/{id}`
- **Description:** Retrieve an assessment by its MongoDB ObjectId.
- **Response (200 OK):** Wrapped in a `ParseResponse` envelope matching `/parse`.

### 3. `GET /assessments`
- **Description:** Query list of assessments filtered by `start_date` and `end_date` metadata.
- **Query Params:** `limit` (default 20), `offset` (default 0), `start_date` (ISO format), `end_date` (ISO format).
- **Response (200 OK):** A list of wrapped assessments.

---

## Pipeline Execution CLI (D5)

Run the real integration pipeline end-to-end on `clinical_assessment.wav` located at the root of the project:
```powershell
.venv\Scripts\python.exe scripts/run_pipeline.py
```
This script will:
1. Locate `clinical_assessment.wav` at the root.
2. Transcribe using the configured backend.
3. Run the LangGraph workflow.
4. Validate confidence metrics.
5. Format and print the final JSON payload.
6. Exit with code `0` on success and non-zero code on failure.

---

## Testing

Run the test suite (all external services mocked for offline reliability):
```powershell
.venv\Scripts\pytest
```

---

## Security, Privacy and Invariants

- **Credential Safety:** Credentials are never printed, logged, or exposed in error tracebacks. FastAPI handlers intercept uncaught exceptions to hide sensitive metadata.
- **Privacy Compliance:** Audio transcript contents containing clinical metadata are never logged in terminal output or persistent logs.
- **Hallucination Prevention:** The extraction LLM is strictly constrained to extract only what is explicit. If it hallucinates values that fail validation (or if the audit assigns low confidence), the request is rejected entirely and nothing is saved to the database.
- **Offline Alternative:** To run offline without sending data to public clouds, set `WHISPER_PROVIDER=local` and `LLM_PROVIDER=ollama`.
