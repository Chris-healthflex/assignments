# Stance Health – Clinical Assessment Pipeline

An automated pipeline that turns a recorded clinician-patient session (WAV audio) into a structured, schema-validated clinical assessment, stored in MongoDB and exposed via a FastAPI service.

## Pipeline Overview

WAV audio
↓
Whisper (speech-to-text)
↓
LangGraph agent (Groq LLM) — clinical entity extraction
↓
FirstAssessment (Pydantic schema) — validation + confidence flagging
↓
MongoDB (Atlas) — persistence
↓
FastAPI — REST endpoints


## Setup

### 1. Clone and create a virtual environment

```bash
git clone <your-repo-url>
cd stance-health-assessment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

Copy `.env.example` to `.env` and fill in your own values:

```bash
copy .env.example .env        # Windows
# cp .env.example .env        # macOS/Linux
```

`.env` requires:   MONGO_URI=your_mongodb_atlas_connection_string
MONGO_DB_NAME=stance_health
GROQ_API_KEY=your_groq_api_key
WHISPER_MODEL_SIZE=base


- **MONGO_URI** — MongoDB Atlas connection string (a free-tier cluster works).
- **GROQ_API_KEY** — free API key from [console.groq.com](https://console.groq.com), used for the LangGraph extraction step.
- **WHISPER_MODEL_SIZE** — Whisper model size (`base` is a good speed/accuracy tradeoff for development).

### 4. Run the test pipeline (D5)

Runs the full pipeline once, end-to-end, against the sample WAV in `data/`, and saves the result to MongoDB:

```bash
python -u tests\test_pipeline.py
```

### 5. Run the API server

```bash
uvicorn app.main:app --reload
```

Interactive API docs: `http://127.0.0.1:8000/docs`

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/assessments/parse` | Upload a WAV file; runs the full pipeline and returns the parsed `FirstAssessment` JSON. Does not save to DB. |
| POST | `/assessments` | Save an already-parsed `FirstAssessment` JSON to MongoDB. |
| GET | `/assessments/{id}` | Retrieve a single saved assessment by its MongoDB ID. |
| GET | `/assessments` | List all saved assessments, optionally filtered by date. |

### Examples

**Parse a WAV file (EP1)**

```bash
curl -X POST "http://127.0.0.1:8000/assessments/parse" \
  -H "accept: application/json" \
  -F "file=@data/clinical_assessment.wav;type=audio/wav"
```

**Save a parsed assessment (EP2)**

```bash
curl -X POST "http://127.0.0.1:8000/assessments" \
  -H "Content-Type: application/json" \
  -d '{
    "clinicalDetails": {"clinicalHistory": "...", "chiefComplaint": "...", "duration": "8 months"},
    "subjectiveAssessments": [],
    "objectiveAssessment": {"tests": []},
    "subjectiveGoals": [],
    "objectiveGoals": [],
    "recommendation": [],
    "patientAdvice": {"adviceDetails": "..."},
    "flaggedFields": []
  }'
```

**Retrieve by ID (EP3)**

```bash
curl -X GET "http://127.0.0.1:8000/assessments/6a96ecb4e96f3bd158b8330d" \
  -H "accept: application/json"
```

**List all, optionally filtered by date (EP4)**

```bash
curl -X GET "http://127.0.0.1:8000/assessments" \
  -H "accept: application/json"

curl -X GET "http://127.0.0.1:8000/assessments?date=2026-09-01" \
  -H "accept: application/json"
```

## Design Decisions

**LLM provider: Groq instead of OpenAI.**
The extraction agent was originally built against OpenAI's `gpt-4o-mini`. During development the OpenAI account ran out of credits, so the extraction LLM was switched to Groq (`openai/gpt-oss-120b`) via `langchain-groq`. Because LangChain's chat-model interface (`.invoke()`) is provider-agnostic, this was a drop-in swap — the LangGraph state machine, prompt, and JSON parsing logic were unaffected.

**Anti-hallucination strategy.**
Whisper transcription is not perfect and can produce garbled or ambiguous words (e.g. mishearing a number or a clinical term). The extraction prompt explicitly instructs the LLM to leave a field empty rather than guess a plausible-sounding value when the transcript is ambiguous. This was validated directly against the provided sample audio: an unclear phrase in the transcript (transcribed as "negic 5" where a knee extension angle was being described) is left blank in the final output rather than being converted into an invented numeric value.

**No fabricated placeholders.**
Earlier versions of the mapping layer inserted an empty placeholder object into list fields (e.g. `subjectiveAssessments`) whenever the extraction agent found nothing to report, to satisfy a "no empty arrays" assumption. This was corrected: if the transcript genuinely contains no subjective assessment, subjective goal, etc., the corresponding list stays a true empty array (`[]`), and the field name is instead recorded in `flaggedFields`.

**Confidence flagging (`flaggedFields`).**
`FirstAssessment` includes a `flaggedFields` array listing which optional fields (e.g. `duration`, `subjectiveGoals`, `objectiveGoals`) came back empty after extraction. This makes low-information sections explicitly visible to a reviewer instead of silently blank, without blocking the assessment from being saved.

**Hard failure vs soft flagging.**
Two fields — `chiefComplaint` and `clinicalHistory` — are treated as critical. If either is empty after extraction, `POST /assessments/parse` returns `422` with the list of missing fields, since an assessment without these core facts isn't usable. All other empty fields are only soft-flagged via `flaggedFields`, since a session may legitimately not cover every category (goals, subjective assessments, etc.).

## Known Limitations

- **Whisper accuracy.** The `base` model trades accuracy for speed. It can misheard clinical terms and numbers (see the anti-hallucination note above) — a larger model (`small`/`medium`) would reduce this but increase latency and local resource usage.
- **Single-speaker transcript.** The transcription step does not perform speaker diarization, so clinician and patient speech are transcribed as one continuous block. Downstream extraction relies on context alone to distinguish reported symptoms from clinician assessment.
- **English only.** Both Whisper (as configured) and the extraction prompt assume English-language input.
- **No retry/backoff on the LLM call.** If the Groq API is rate-limited or briefly unavailable, `extract_clinical_data` raises immediately rather than retrying — acceptable for this assignment's scope, but a production version would add retry logic.
- **`flaggedFields` is field-level, not value-level.** A field is only flagged when it is fully empty. A confidently-stated but *clinically implausible* value (e.g. a wildly out-of-range angle) is not currently flagged — only genuinely missing/ambiguous information is caught.

## Tech Stack

- **Transcription:** OpenAI Whisper (local, `base` model)
- **Extraction agent:** LangGraph + Groq (`openai/gpt-oss-120b`) via `langchain-groq`
- **Schema/validation:** Pydantic v2
- **Database:** MongoDB Atlas (via `pymongo`)
- **API:** FastAPI + Uvicorn

