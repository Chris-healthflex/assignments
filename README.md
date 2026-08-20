# Clinical Audio to Structured FirstAssessment Pipeline

A Python backend and clinical information extraction pipeline that ingests clinician-patient WAV audio recordings, transcribes them using Whisper, extracts structured clinical information via a LangGraph extraction agent adhering to strict evidence-grounding constraints, validates the output against the Pydantic v2 `FirstAssessment` schema, and provides persistence via MongoDB and REST APIs via FastAPI.

---

## Architecture and Pipeline Flow

```
[ WAV Audio File Upload ]
           |
           v
[ POST /assessments/parse ]
           |
           v
[ Audio Validator (RIFF header, channels, size) ]
           |
           v
[ Whisper Transcriber (API / Local / Mock) ]
           |
           v
[ Plain Text Consultation Transcript ]
           |
           v
[ LangGraph Extraction Agent ]
  - validate_transcript
  - extract_clinical_entities
  - validate_extraction (anti-hallucination & grounding)
  - build_first_assessment
           |
           v
[ Pydantic v2 FirstAssessment Schema ]
     /                   \
    v                     v
[ MongoDB Storage ]   [ HTTP JSON Response ]
```

---

## Anti-Hallucination and Grounding Strategy

Clinical information extraction requires strict adherence to transcript evidence:

1. **System Extraction Prompting**: The system prompt strictly prohibits inferring unstated symptoms, diagnosing without clinician mention, converting subjective complaints into synthetic test names, or inventing medication dosages.
2. **Speaker-Attributed Evidence Grounding**:
   - Recommendations and patient advice are validated strictly against the Doctor transcript.
   - Clinical history, chief complaints, and durations are validated against the Patient transcript.
3. **Auto-Pruning of Unsupported Items**:
   - Any extracted recommendation or assessment item lacking direct transcript evidence is automatically pruned before final schema construction.
4. **Pydantic v2 Schema Guarantees**:
   - Missing fields serialize as empty strings `""`, never `null`.
   - List fields remain JSON arrays even when empty or containing single items.
   - Disallows unexpected/extra keys (`extra="forbid"`).

---

## FirstAssessment Schema

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

---

## API Endpoints

The FastAPI server exposes 4 primary endpoints under `/assessments`:

1. **EP1: `POST /assessments/parse`**
   - Ingests a single consultation WAV audio recording (`multipart/form-data`).
   - Validates audio, transcribes via Whisper, runs LangGraph extraction, and returns structured `FirstAssessment` JSON.

2. **EP1-Dual: `POST /assessments/parse-dual`**
   - Ingests separate `doctor_file` and `patient_file` WAV audio streams.
   - Transcribes both tracks and runs speaker-aware extraction.

3. **EP2: `POST /assessments`**
   - Accepts a valid `FirstAssessment` JSON payload.
   - Persists the record to MongoDB and returns the generated UUID and timestamp.

4. **EP3: `GET /assessments/{id}`**
   - Retrieves a stored assessment document by its UUID.
   - Returns HTTP 404 if the ID does not exist.

5. **EP4: `GET /assessments`**
   - Lists stored assessment documents with optional creation date filtering (e.g., `?date=2026-08-20`).

Interactive Swagger documentation is available at `http://127.0.0.1:8000/docs`.

---

## Setup and Installation

### Prerequisites
- Python 3.10+ (tested on Python 3.12)
- MongoDB running locally on `mongodb://localhost:27017` (or remote URI)
- Groq or OpenAI API Key for Whisper and LLM extraction

### Installation Steps

1. **Clone the repository and create a virtual environment**:
   ```bash
   git clone https://github.com/Stance-Health/ai-assignments.git
   cd ai-assignments
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Environment Variables**:
   Copy `.env.example` to `.env` and fill in your API credentials:
   ```bash
   cp .env.example .env
   ```

   Example `.env`:
   ```env
   HOST=0.0.0.0
   PORT=8000
   MONGODB_URI=mongodb://localhost:27017
   MONGODB_DATABASE=clinical_assessments
   MONGODB_COLLECTION=assessments

   GROQ_API_KEY=your_groq_api_key_here
   LLM_BASE_URL=https://api.groq.com/openai/v1
   LLM_MODEL=openai/gpt-oss-120b
   LLM_TEMPERATURE=0.0

   WHISPER_MODE=openai
   WHISPER_MODEL=whisper-large-v3-turbo
   CONFIDENCE_THRESHOLD=0.70
   ```

---

## Running the Application

### Start the FastAPI Dev Server
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```
The API is live at `http://127.0.0.1:8000` with Swagger UI at `http://127.0.0.1:8000/docs`.

---

## Verification and Testing

### 1. Run Automated Test Suite
Execute all 39 unit, integration, and guardrail tests:
```bash
pytest -v
```

### 2. Run End-to-End Pipeline on Audio Files
Run the pipeline directly from the command line on WAV recordings:
```bash
python scripts/test_pipeline.py data/day1_consultation01_doctor.wav data/day1_consultation01_patient.wav
```
The script will validate the WAV files, transcribe the doctor and patient tracks with Whisper, run LangGraph extraction with grounding checks, and persist the record to MongoDB.
