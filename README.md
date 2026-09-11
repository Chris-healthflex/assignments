# Clinical Assessment Voice Form Filler

> **Turn a clinical audio session into a structured assessment report — automatically.**

A production-ready **FastAPI** microservice that accepts a WAV recording of a real clinician-patient session, transcribes it with **OpenAI Whisper** (via Groq or local CPU), extracts every clinical field using a **5-node LangGraph** agent pipeline, validates extraction confidence field-by-field, and persists the result to **MongoDB Atlas** — all with strict **Pydantic v2** schema enforcement.

---

## Pipeline Architecture

```
[WAV Upload] --> [WAV Validator] --> [Whisper STT] --> [Transcript]
                                                            |
                                          +----- LangGraph StateGraph -----+
                                          |                                |
                                     extract_node                         |
                                     (LLM + 3 retries)                    |
                                          |                                |
                                     normalize_node                        |
                                     (dates, units)                        |
                                          |                                |
                                     confidence_node                       |
                                     (LLM field audit)                     |
                                          |                                |
                                     validate_node                         |
                                     (< 0.70 -> 422)                      |
                                          |                                |
                                          +--------------------------------+
                                          |
                             [FirstAssessment JSON / 422 detail]
                                          |
                                    [MongoDB Atlas]
                               (via POST /assessments)
```

---

## Key Features

| Feature | Detail |
|---|---|
| **Double-Safe WAV Validation** | Extension check AND binary `RIFF`/`WAVE` magic-byte header validation |
| **Dual Transcription Backend** | `WHISPER_PROVIDER=groq` (remote `whisper-large-v3-turbo`) or `local` (offline CPU via `faster-whisper`) |
| **Self-Healing LLM Extraction** | Up to 3 retry attempts, feeding Pydantic validation errors back to the model before falling back |
| **Automated 429 Rate-Limit Handling** | Exponential backoff up to 5 attempts on free-tier token-limit errors |
| **Strict Schema Invariants** | No `null` values; strings default `""`, arrays default `[]`; `extra="forbid"` enforced |
| **Granular Confidence Auditing** | LLM scores each populated field vs raw transcript — below `0.70` returns `422` with field + reason + score |
| **Hallucination Prevention** | Extraction prompt explicitly forbids inventing values; low-confidence results are blocked from DB |
| **MongoDB Atlas Integration** | Thread-safe singleton via `lifespan`; `created_at`/`updated_at` metadata kept off public responses |
| **Offline Mode** | Set `WHISPER_PROVIDER=local` + `LLM_PROVIDER=ollama` for fully air-gapped operation |

---

## Project Structure

```
assignments/
+-- app/
|   +-- api/
|   |   +-- routes/
|   |       +-- assessments.py     # All 4 REST endpoints
|   +-- core/
|   |   +-- config.py              # Pydantic Settings (env-driven)
|   +-- database/
|   |   +-- mongodb.py             # Singleton MongoClient + CRUD layer
|   +-- graph/
|   |   +-- assessment_graph.py    # 5-node LangGraph StateGraph
|   |   +-- state.py               # AssessmentState TypedDict
|   +-- models/
|   |   +-- assessment.py          # FirstAssessment Pydantic v2 schema
|   +-- schemas/
|   |   +-- assessment.py          # ParseResponse, ValidationErrorDetail
|   +-- services/
|   |   +-- confidence.py          # LLM field-level confidence auditor
|   |   +-- extraction.py          # LLM extraction pipeline + fallback
|   |   +-- transcription.py       # Whisper STT (Groq / local)
|   +-- main.py                    # FastAPI app + lifespan + exception handlers
+-- scripts/
|   +-- run_pipeline.py            # CLI: end-to-end run on clinical_assessment.wav
+-- tests/
|   +-- conftest.py                # pytest fixtures (TestClient, mock Mongo)
|   +-- test_assessments.py        # 12 unit tests (all mocked, offline-safe)
+-- clinical_assessment.wav        # Provided input WAV file
+-- .env.example                   # Environment variable template
+-- requirements.txt
+-- README.md
```

---

## FirstAssessment Schema

The output JSON must **exactly** match this structure. No extra fields, no renamed keys, no `null` values.

```json
{
  "clinicalDetails": {
    "clinicalHistory": "string",
    "chiefComplaint":  "string",
    "duration":        "string"
  },
  "subjectiveAssessments": [
    { "testName": "string", "conclusion": "string" }
  ],
  "objectiveAssessment": {
    "tests": [
      {
        "testName":  "string",
        "unitName":  "string",
        "value":     "string",
        "left":      "string",
        "right":     "string",
        "comments":  "string"
      }
    ]
  },
  "subjectiveGoals": [
    { "goalDetails": "string", "targetDate": "string" }
  ],
  "objectiveGoals": [
    {
      "goalName":     "string",
      "goalCategory": "string",
      "unitName":     "string",
      "value":        "string",
      "targetDate":   "string"
    }
  ],
  "recommendation": [
    { "sessionType": "string", "sessionFrequency": "string" }
  ],
  "patientAdvice": {
    "adviceDetails": "string"
  }
}
```

> All string fields default to `""`. All array fields default to `[]`. `extra="forbid"` is enforced on every model.

---

## REST API Endpoints

### `POST /assessments/parse`

Transcribes a WAV upload and extracts the `FirstAssessment` JSON. Does **not** persist — returns the parsed structure directly.

- **Payload:** `multipart/form-data` with key `file` (.wav only)
- **Success `200 OK`:**
  ```json
  {
    "clinicalDetails": { "clinicalHistory": "...", "chiefComplaint": "...", "duration": "..." },
    "subjectiveAssessments": [],
    "objectiveAssessment": {
      "tests": [{ "testName": "Knee flexion", "unitName": "degrees", "left": "124", "right": "130", "value": "", "comments": "" }]
    },
    "subjectiveGoals": [],
    "objectiveGoals": [],
    "recommendation": [{ "sessionType": "Physiotherapy", "sessionFrequency": "once weekly for 4 sessions" }],
    "patientAdvice": { "adviceDetails": "" }
  }
  ```
- **Confidence Failure `422 Unprocessable Entity`:**
  ```json
  {
    "detail": [
      {
        "field":      "clinicalDetails.duration",
        "reason":     "Transcript mentions knee pain but does not state 8 months",
        "confidence": 0.35
      }
    ]
  }
  ```
- **`400 Bad Request`:** File extension is not `.wav`, file is empty, or binary RIFF/WAVE header is invalid.

---

### `POST /assessments`

Saves a pre-validated `FirstAssessment` JSON body to MongoDB Atlas.

- **Payload:** `application/json` body matching the `FirstAssessment` schema
- **Success `201 Created`:**
  ```json
  {
    "id": "66c72b2f9b1d8b2d88888888",
    "assessment": { "clinicalDetails": { ... }, ... }
  }
  ```
- **`503`:** MongoDB Atlas is unreachable.

---

### `GET /assessments/{id}`

Retrieves a saved assessment by its MongoDB ObjectId string.

- **Success `200 OK`:** `ParseResponse` envelope — `{ "id": "...", "assessment": { ... } }`
- **`400`:** Invalid ObjectId format
- **`404`:** Assessment not found

---

### `GET /assessments`

Lists assessments sorted by `created_at` descending. Supports date range filtering.

| Query Param  | Type           | Default | Description                    |
|---|---|---|---|
| `start_date` | ISO date string | —      | Filter: created after this date |
| `end_date`   | ISO date string | —      | Filter: created before this date |
| `limit`      | int (1–100)    | `20`    | Page size                      |
| `offset`     | int (>=0)      | `0`     | Page offset                    |

- **Success `200 OK`:** Array of `ParseResponse` objects.

---

### `GET /health`

```json
{ "status": "ok", "database": "connected" }
```

Returns `"degraded"` if MongoDB is unreachable (app still boots).

---

## Environment Variables (`.env`)

Copy `.env.example` to `.env` and fill in your credentials:

```ini
# MongoDB Atlas
MONGODB_URI=mongodb+srv://<user>:<password>@cluster.mongodb.net/?appName=clinical-assessment
MONGODB_DATABASE=clinical_assessment

# LLM Extraction (Groq recommended, or Ollama for offline)
LLM_PROVIDER=groq
LLM_MODEL=llama-3.3-70b-versatile
GROQ_API_KEY=gsk_your_key_here
GROQ_BASE_URL=https://api.groq.com/openai/v1
OLLAMA_BASE_URL=http://localhost:11434

# Audio Transcription (Whisper via Groq or local CPU)
WHISPER_PROVIDER=groq
WHISPER_MODEL=whisper-large-v3-turbo
WHISPER_DEVICE=cpu
WHISPER_COMPUTE_TYPE=int8

# Confidence Gate
CONFIDENCE_THRESHOLD=0.70
```

---

## Installation & Setup

### 1. Clone the repository

```powershell
git clone https://github.com/Chris-healthflex/assignments
cd assignments
```

### 2. Create and activate a virtual environment

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
pip install -r requirements.txt
```

For the **offline/local Whisper** fallback:

```powershell
pip install faster-whisper
```

### 4. Configure environment

```powershell
Copy-Item .env.example .env
# Edit .env with your MONGODB_URI and GROQ_API_KEY
```

### 5. Start the FastAPI server

```powershell
.venv\Scripts\uvicorn app.main:app --reload --port 8000
```

Open the **interactive Swagger UI** at: `http://127.0.0.1:8000/docs`

---

## Testing

Run the full test suite — all 12 tests run **completely offline** (all external services are mocked):

```powershell
.venv\Scripts\pytest -v
```

Expected output:

```
tests/test_assessments.py::test_health_check                    PASSED
tests/test_assessments.py::test_parse_invalid_extension         PASSED
tests/test_assessments.py::test_parse_empty_file                PASSED
tests/test_assessments.py::test_parse_bad_wav_header            PASSED
tests/test_assessments.py::test_parse_success                   PASSED
tests/test_assessments.py::test_save_assessment_success         PASSED
tests/test_assessments.py::test_parse_low_confidence            PASSED
tests/test_assessments.py::test_get_assessment_success          PASSED
tests/test_assessments.py::test_get_assessment_invalid_id       PASSED
tests/test_assessments.py::test_get_assessment_not_found        PASSED
tests/test_assessments.py::test_list_assessments_success        PASSED
tests/test_assessments.py::test_fallback_validate_logic         PASSED

12 passed in ~1.2s
```

---

## Pipeline CLI (D5 Deliverable)

Run the end-to-end pipeline on the provided `clinical_assessment.wav`:

```powershell
.venv\Scripts\python.exe scripts/run_pipeline.py
```

This script:
1. Locates `clinical_assessment.wav` at the project root
2. Validates the WAV header and transcribes using the configured backend
3. Runs the full 5-node LangGraph extraction workflow
4. Validates all field confidence scores against the threshold
5. Prints the final `FirstAssessment` JSON to stdout
6. Exits `0` on success, non-zero on each failure stage

---

## Design Decisions

### Why LangGraph instead of a plain LLM call?

LangGraph gives us a **deterministic, auditable execution graph** with explicit node boundaries. Each stage (`extract -> normalize -> confidence -> validate`) can fail, retry, or skip independently without polluting the other stages. The `StateGraph` compilation makes the entire pipeline reproducible and extensible — adding a `human-in-the-loop` review node later is one `add_node` call.

### Why 3-attempt extraction with feedback injection?

LLMs occasionally produce structurally invalid JSON or miss required schema fields, especially under free-tier token pressure. By feeding the exact `ValidationError` message back as a follow-up prompt, the model self-corrects without a full restart. A `fallback_validate` safety net handles the case where all three retries fail by parsing whatever partial data is available field-by-field, ensuring the pipeline always returns a complete, schema-valid output.

### Why a separate `confidence_node` after extraction?

Clinical data is safety-critical. Simply extracting a plausible-looking value is insufficient — a separate LLM **audit pass** comparing each populated field against the raw transcript catches silent hallucinations (values that *look* valid but were never actually spoken). This two-stage approach (extract, then audit) is significantly more reliable than asking a single prompt to both extract and self-grade.

### Why separate `POST /assessments/parse` and `POST /assessments`?

Per the assignment spec, `/assessments/parse` is the **extraction** endpoint and `POST /assessments` is the **persistence** endpoint. Keeping them separate follows single-responsibility design — the caller can inspect parsed output before committing it to the database, naturally supporting a human-review or client-confirmation workflow.

### Why exponential backoff for rate-limit handling?

Free-tier Groq API has strict tokens-per-minute limits. The `invoke_llm_with_retry` wrapper catches `429` errors and backs off with doubling delays (`4s -> 8s -> 16s -> 32s -> 64s`) rather than immediately failing, making the full pipeline usable without a paid API tier.

### MongoDB singleton via `lifespan`

Using FastAPI's `lifespan` context manager (rather than a `startup` event) ensures the MongoClient is properly initialised before any request is served and cleanly closed on process exit — avoiding connection leaks in hot-reload development.

---

## Security & Privacy

- **No credentials in logs:** All exception handlers strip sensitive metadata from error responses.
- **No transcript logging:** Clinical audio content is never written to log files or terminal output.
- **Hallucination prevention:** The extraction LLM is strictly instructed to emit only values explicitly present in the transcript. Low-confidence fields trigger a `422` rejection — nothing is written to MongoDB.
- **Offline mode:** Set `WHISPER_PROVIDER=local` and `LLM_PROVIDER=ollama` for complete air-gapped operation with no data sent to external APIs.

---

## Deliverables Checklist

| # | Item | Status |
|---|---|---|
| D1 | FastAPI service — all 4 endpoints working | Done |
| D2 | Whisper transcription module (WAV to text) | Done |
| D3 | LangGraph agent with FirstAssessment Pydantic output | Done |
| D4 | MongoDB models + connection + save/retrieve logic | Done |
| D5 | Test script: run pipeline on provided WAV, print JSON | Done |
| D6 | README — setup instructions + design decisions | Done |

---

## Swagger UI

Once running, visit `http://127.0.0.1:8000/docs` to explore and test all endpoints interactively.

---

*Built with FastAPI, LangGraph, Groq Whisper, MongoDB Atlas, and Pydantic v2*
