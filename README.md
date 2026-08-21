# Clinical Assessment Service

Production-aware pipeline that transcribes a real clinician-patient audio session and extracts a fully structured `FirstAssessment` JSON — exactly matching the schema expected by the production frontend.

## Features

- Upload WAV audio → structured clinical assessment
- Local Whisper transcription (faster-whisper)
- Groq LLM extraction via LangGraph
- Strict Pydantic v2 schema (`extra="forbid"`)
- Anti-hallucination guardrails with fuzzy source verification
- Field-level confidence gate returning HTTP 422 on low confidence
- MongoDB persistence with 4 REST endpoints
- Full test suite including end-to-end on provided WAV

---

## Architecture

```
WAV upload (multipart)
        │
        ▼
[1] File validation (MIME, size, WAV header)
        │
        ▼
[2] Audio conversion → 16 kHz mono PCM (pydub + imageio-ffmpeg)
        │
        ▼
[3] Whisper transcription (faster-whisper, thread-pool offloaded)
        │
        ▼
[4] LangGraph extraction agent
        │
        ├─ nodes: clinical_details, subjective_assessments,
        │         objective_assessment, goals, recommendation, patient_advice
        ├─ each node returns: value, is_mentioned, confidence, source_quote
        ├─ conditional retry edge (max 1 retry on low confidence)
        │
        ▼
[5] Fuzzy source verification + confidence aggregation
        │
        ▼
[6] Pydantic v2 strict assembly → FirstAssessment
        │
        ▼
[7] Fail-closed gate
        │
   confidence OK? ─── no ──→ HTTP 422 field-level errors
        │
       yes
        │
        ▼
   200 FirstAssessment JSON
        │
        ▼
   POST /assessments → MongoDB
```

---

## Tech Stack

- Python 3.10+ / FastAPI
- faster-whisper (local transcription)
- LangGraph + LangChain Groq (structured extraction)
- Pydantic v2 (strict schema)
- MongoDB (Motor async driver)
- pytest + mongomock-motor (testing)

---

## Project Structure

```
app/
├── main.py           # FastAPI app factory
├── api/               # HTTP routes (thin)
├── core/               # config, logging, exceptions
├── schemas/            # Pydantic models (public + internal)
├── audio/               # validation, preprocessing, transcription
├── agent/                # LangGraph graph, nodes, prompts, LLM client
├── guardrails/            # source match, validators, confidence gate
├── db/                     # MongoDB client, models, repository
└── services/                # orchestration (parse_service.py)
tests/
├── unit/              # schema, guardrails, audio, agent nodes
├── integration/        # API endpoints, confidence gate
└── e2e/                  # real WAV pipeline (D5)
scripts/
├── run_pipeline.py    # CLI wrapper for e2e
docs/
├── architecture.md    # detailed architecture and design decisions
└── golden_output.json  # verified output from provided WAV
```

---

## Setup Instructions

### Prerequisites

- Python 3.10+
- MongoDB (via Docker or local install)
- Groq API key
- ffmpeg (optional if using `imageio-ffmpeg`; already included)

### 1. Clone and enter repo

```bash
git clone https://github.com/Chris-healthflex/assignments
cd assignments
git checkout candidate/mishrasundram091-gmail-com-cfdf
```

### 2. Create virtual environment & install dependencies

```bash
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows

pip install -r requirements.txt
```

### 3. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` and set:

- `GROQ_API_KEY` — your Groq API key
- `GROQ_MODEL` — `openai/gpt-oss-120b` (default used for this assignment)
- `MONGO_URI` — MongoDB connection string (default `mongodb://localhost:27017`)
- Optionally adjust `WHISPER_MODEL` (default `small`) and `CONFIDENCE_THRESHOLD` (default `0.75`)

### 4. Start MongoDB

```bash
docker-compose up -d
```

Or use a local MongoDB service.

### 5. Run the FastAPI service

```bash
uvicorn app.main:app --reload
```

Service runs at `http://127.0.0.1:8000`.

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/assessments/parse` | Upload WAV, returns `FirstAssessment` JSON (or 422) |
| POST | `/api/v1/assessments` | Save parsed assessment to MongoDB |
| GET | `/api/v1/assessments/{id}` | Retrieve assessment by ID |
| GET | `/api/v1/assessments` | List all, optional `start_date` / `end_date` filters |

---

## Example Output

Running the pipeline on the provided `clinical_assessment.wav` produces the following JSON (exact schema):

```json
{
  "clinicalDetails": {
    "clinicalHistory": "Patient was normal eight months ago, then involved in a road traffic accident resulting in left tibial condy fracture and avulsion ACL tear; underwent open reduction and internal fixation, followed by 4-6 weeks non-weight bearing and progressive loading.",
    "chiefComplaint": "left knee pain, difficulty performing functional activities and difficulty walking along with ankle and back pain during prolonged walking following surgery",
    "duration": "8 months"
  },
  "subjectiveAssessments": [
    {
      "testName": "Presenting symptoms",
      "conclusion": "Left knee pain, difficulty performing functional activities and difficulty walking along with ankle and back pain during prolonged walking"
    },
    {
      "testName": "Pain characteristics",
      "conclusion": "Moderate pain with mild irritability particularly during prolonged walking and standing, relieved with rest"
    }
  ],
  "objectiveAssessment": {
    "tests": [
      {
        "testName": "Knee flexion",
        "unitName": "degrees",
        "value": "",
        "left": "124",
        "right": "130",
        "comments": ""
      },
      {
        "testName": "Knee extension",
        "unitName": "degrees",
        "value": "",
        "left": "20",
        "right": "5",
        "comments": ""
      },
      {
        "testName": "Hip internal rotation",
        "unitName": "degrees",
        "value": "",
        "left": "45",
        "right": "45",
        "comments": ""
      },
      {
        "testName": "Hip external rotation",
        "unitName": "degrees",
        "value": "",
        "left": "60",
        "right": "60",
        "comments": ""
      },
      {
        "testName": "Ankle dorsiflexion",
        "unitName": "degrees",
        "value": "",
        "left": "4.5",
        "right": "12",
        "comments": ""
      }
    ]
  },
  "subjectiveGoals": [],
  "objectiveGoals": [],
  "recommendation": [
    {
      "sessionType": "Physiotherapy",
      "sessionFrequency": "once weekly for four sessions"
    }
  ],
  "patientAdvice": {
    "adviceDetails": "Physiotherapy was recommended once weekly for four sessions, with emphasis on restoring the extension, improving knee stability and single leg stability, strengthening the quadriceps and functional lower limb musculature, improving ankle mobility, and activating the posterior chain."
  }
}
```

This output is also saved in `docs/golden_output.json`.

---

## Running Tests

### Unit & integration tests

```bash
pytest tests/unit tests/integration
```

All 13 tests pass.

### End-to-end pipeline on provided WAV

```bash
python tests/e2e/test_pipeline.py
```

This downloads the WAV (if not present), transcribes it, extracts the structured assessment, and prints the final JSON.

You can also run:

```bash
python scripts/run_pipeline.py
```

---

## Design Decisions

### 1. Exact schema with Pydantic v2

`FirstAssessment` uses `extra="forbid"` and all fields have defaults. Arrays default to `[]`, strings to `""`. This guarantees the output matches the production frontend contract — no extra keys, no null values.

### 2. Separation of public and internal models

The public `FirstAssessment` model is never polluted with extraction metadata. Internal models (`ExtractionResult`, `ExtractionField`) carry `confidence`, `is_mentioned`, and `source_quote`. The API returns only the public schema.

### 3. Anti-hallucination guardrails

Every non-empty extracted value must have a source quote and a confidence score. A fuzzy source-match (rapidfuzz) checks whether the value appears in the transcript. If a value cannot be supported, its confidence is reduced. Values below the threshold are rejected with HTTP 422. The LLM is instructed to return `is_mentioned=false` for absent fields, which distinguishes "verified absent" from "low confidence present".

### 4. LLM Used

We use Groq's **`openai/gpt-oss-120b`** via LangChain's `ChatGroq`. The model is instructed with a system prompt that prohibits inference and requires source quotes for every field. Temperature is set to `0.0` for deterministic output.

### 5. Confidence threshold and retry

`CONFIDENCE_THRESHOLD` (default `0.75`) is applied to every non-empty field. If any field falls below, the LangGraph agent retries once from the beginning, then the pipeline fails closed with a 422 and field-level error details.

### 6. Fuzzy source matching

Exact substring matching fails with ASR errors and natural language variations (e.g., "3 days" vs "three days"). We use fuzzy partial ratio with normalization (numbers to words, lowercasing) and adjust confidence gradually rather than hard-gating.

### 7. Short medical units

Common medical units (e.g., cm, degrees, mmHg) are short and often transcribed inconsistently. We treat them specially in source matching to avoid unfair confidence penalties when the unit is standard.

### 8. Thread-pool offloading

Whisper transcription is CPU-bound. It runs in a thread pool to keep the FastAPI event loop responsive.

### 9. Deliberate scope cuts (documented)

The following were intentionally not implemented for this assignment, but are listed as next steps:

- Diarization (speaker labels) — not required for schema extraction.
- VAD segmentation — single audio file fits in memory.
- Idempotency hashing — parse endpoint is stateless.
- Correlation-ID logging — not needed for this assignment's evaluation.
- Retry/backoff for external LLM — Groq call is treated as reliable.
- Tracing/metrics — out of scope for now.

---

## Security Note

- Never commit `.env` or any real API keys.
- The `.gitignore` excludes `.env`, `.wav`, caches, and virtual environments.
- Rotate your Groq API key if it has been exposed.

---

## Troubleshooting

**Couldn't find ffmpeg or avconv**
Install ffmpeg or rely on `imageio-ffmpeg`. The code explicitly sets `AudioSegment.converter` to the bundled binary, so the warning is harmless.

**pytest fixture not found**
Ensure `conftest.py` is in the project root (not inside `tests/`) and that `tests/` does not contain `__init__.py`.

**Groq API errors**
Verify `GROQ_API_KEY` and `GROQ_MODEL` in `.env`. This project uses `openai/gpt-oss-120b` by default. If it's unavailable, try `llama-3.3-70b-versatile` or `llama-3.1-70b-versatile` as fallbacks.

---

## Deliverables Checklist

- ☑ D1: FastAPI service with 4 endpoints
- ☑ D2: Whisper transcription module
- ☑ D3: LangGraph agent with Pydantic output
- ☑ D4: MongoDB models + repository
- ☑ D5: Test script runs on provided WAV
- ☑ D6: README with setup and design decisions