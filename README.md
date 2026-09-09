# Voice/Note → Structured Clinical Assessment Form Filler

A robust clinical audio processing pipeline that transcribes clinician–patient consultation recordings (WAV), extracts clinical findings, and produces structured JSON conforming strictly to Stance Health's production `FirstAssessment` schema with MongoDB persistence.

```text
[Clinical Audio WAV] ──► [Whisper ASR (soundfile, in-process)]
                                       │
                                       ▼ Transcript
                             [LangGraph Clinical Agent]
                                ├── Extract (Structured clinical entities)
                                ├── Normalize (Strict Pydantic v2 coercion: null -> "")
                                └── Audit (Anti-hallucination guardrails on ROM/numbers)
                                       │
                                       ▼ FirstAssessment JSON (Exact Schema)
                             [FastAPI Service] ──► [MongoDB Persistence]
```

---

## Key Highlights & Design Decisions

1. **Exact-Match Schema Conformance (`FirstAssessment`)**:
   - Built with Pydantic v2: enforces `extra="forbid"`, null strings coerced to `""`, and all lists guaranteed to be lists.
   - The JSON payload returned matches the exact keys expected by the frontend without extra metadata leaking into the response body.
   - Extraction confidence metrics and flagged issues are delivered via HTTP response headers (`X-Extraction-Confidence`, `X-Extraction-Flags`) and stored in the database's metadata audit trail.

2. **Deterministic Anti-Hallucination Audit**:
   - Spoken numeric values in objective measurements (e.g. range of motion degrees, pain scales) are mechanically verified against the transcript. If a numeric value in the assessment is not substantiated by the transcript, it is flagged with low confidence.
   - Core clinical fields (`chiefComplaint`, `clinicalHistory`) are verified. Missing core fields or confidence below the threshold automatically triggers an HTTP `422 Unprocessable Entity` with a detailed field breakdown.

3. **FFmpeg-Free Audio Processing**:
   - Decodes WAV recordings in-memory using `soundfile` and resamples to 16 kHz mono using `scipy.signal.resample_poly`, removing any external `ffmpeg` binary system dependency.
   - Primed with clinical vocabulary (`CLINICAL_PROMPT`) to maximize medical term accuracy.

4. **Multi-Backend Extractor**:
   - Supports LLM providers (Google Gemini, OpenAI, Anthropic, Ollama) via LangChain / LangGraph structured output, with an integrated rule-based clinical parser fallback to guarantee zero downtime even offline.

---

## Project Structure

```text
assignments/
├── app/
│   ├── __init__.py
│   ├── config.py              # Environment configuration & Pydantic settings
│   ├── schemas.py             # FirstAssessment Pydantic v2 schemas & internal models
│   ├── transcription.py       # Whisper audio loader & transcription engine
│   ├── agent.py               # LangGraph extraction, normalization & audit pipeline
│   ├── db.py                  # MongoDB async client & persistence repository
│   └── main.py                # FastAPI REST service & endpoints
├── scripts/
│   └── run_pipeline.py        # Standalone CLI runner for audio files
├── tests/
│   ├── __init__.py
│   ├── test_schema.py         # Offline schema conformance & guardrail unit tests
│   └── test_api.py            # API endpoint integration tests
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Setup & Installation

### 1. Environment Setup
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure Environment Variables
Copy `.env.example` to `.env` and adjust settings as needed:
```bash
cp .env.example .env
```

---

## Running the Pipeline (CLI)

You can run the end-to-end pipeline directly on any WAV file:

```bash
# Full pipeline: transcribe -> extract -> audit -> output JSON
python scripts/run_pipeline.py clinical_assessment.wav

# Optional: save directly to MongoDB
python scripts/run_pipeline.py clinical_assessment.wav --save

# Optional: transcribe only
python scripts/run_pipeline.py clinical_assessment.wav --transcript-only
```

Outputs:
- Structured JSON printed to `stdout` and saved to `output_assessment.json`.
- Full transcript saved to `transcript.txt`.
- Flagged fields and confidence report printed to `stderr`.

---

## Running the FastAPI Service

Start the development server:

```bash
uvicorn app.main:app --reload --port 8000
```

- **Interactive Swagger Documentation**: `http://127.0.0.1:8000/docs`
- **ReDoc Documentation**: `http://127.0.0.1:8000/redoc`

### API Endpoints

| Method | Path | Description | Status Codes |
|---|---|---|---|
| `GET` | `/health` | Service health check | 200 |
| `POST` | `/assessments/parse` | Upload WAV audio (`file`) → extracts `FirstAssessment` JSON | 200, 400, 422, 500 |
| `POST` | `/assessments` | Save `FirstAssessment` document to MongoDB | 201 Created |
| `GET` | `/assessments/{id}` | Retrieve saved assessment by MongoDB ID | 200, 404 |
| `GET` | `/assessments` | List assessments (supports `from`, `to`, `limit`, `skip`) | 200, 400 |

### Example cURL Requests

#### 1. Transcribe & Extract Audio
```bash
curl -X POST "http://127.0.0.1:8000/assessments/parse" \
  -F "file=@clinical_assessment.wav" \
  -F "save=true"
```

#### 2. Create Assessment Directly
```bash
curl -X POST "http://127.0.0.1:8000/assessments" \
  -H "Content-Type: application/json" \
  -d @output_assessment.json
```

#### 3. Retrieve Assessment
```bash
curl "http://127.0.0.1:8000/assessments/<ID>"
```

#### 4. Filter Assessments by Date
```bash
curl "http://127.0.0.1:8000/assessments?from=2026-09-01T00:00:00Z"
```

---

## Testing

Run the comprehensive test suite with pytest:

```bash
pytest tests/ -v
```

All schema rules, extra key rejections, null coercions, hallucination flags, and API endpoint lifecycles are validated.
