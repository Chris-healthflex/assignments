# Stance Health Clinical Assessment API

A FastAPI processing pipeline that converts local physiotherapy clinical WAV recordings into structured, validated `FirstAssessment` JSON records.

## Tech Stack

* **Python 3.10+** - Core runtime environment.
* **FastAPI** - Web framework and API routing.
* **OpenAI Whisper** - Local speech-to-text engine.
* **LangGraph** - Workflow graph orchestration.
* **Ollama** - Local LLM inference framework.
* **Pydantic v2** - Data validation and schema serialization.
* **MongoDB & PyMongo** - NoSQL document storage.
* **Pytest** - Test execution suite.

---

## Pipeline Architecture & Data Flow

```text
[WAV Audio]
   │
   ▼
[Whisper Transcription]
   │
   ▼
[Raw Transcript]
   │
   ▼
[LangGraph Extraction]
   ├─► [Extract Node] ──► Local Ollama LLM
   └─► [Validate Node]
         │
         ▼
   [FirstAssessment + Confidence]
         │
         ▼
   [Confidence Validation]
         ├─► Confidence below threshold ──► 422 Unprocessable Entity
         └─► Valid assessment
               ├─► Returned by API
               └─► Saved to MongoDB
```

---

## Design Decisions & Architectural Rationale

### 1. Local Processing for Clinical Data
* **Decision:** Speech transcription uses local OpenAI Whisper; structured extraction uses a local Ollama model.
* **Rationale:** Keeps clinical audio and transcripts within the local environment to protect patient privacy and meet strict data compliance standards.

### 2. LangGraph Extraction Workflow
* **Decision:** Implemented as a LangGraph workflow with an extraction node followed by a validation node.
* **Rationale:** Provides a deterministic, stateful processing flow from raw text extraction to formal schema validation.

### 3. Confidence-Based Extraction Validation
* **Decision:** Extracted fields include AI-generated confidence scores checked against a `CONFIDENCE_THRESHOLD`.
* **Rationale:** Prevents clinical hallucinations by automatically rejecting low-confidence extractions with a `422` response.

### 4. Pydantic Schema Validation
* **Decision:** `FirstAssessment` serves as the strict production schema applied immediately post-extraction.
* **Rationale:** Ensures LLM outputs are programmatically cleaned, typed, and structured before API delivery or persistence.

---

## Setup Instructions

### 1. Create and Activate Virtual Environment
```bash
python -m venv venv

# Windows (PowerShell)
.\venv\Scripts\Activate.ps1

# Linux/macOS
source venv/bin/activate
```

### 2. Install Project Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure System Environment
Create a `.env` file in the project workspace root:
```env
APP_NAME="Stance Health Clinical Assessment API"
APP_VERSION="1.0.0"
MONGODB_URI="mongodb://localhost:27017"
MONGODB_DATABASE="stance_assessment"
OLLAMA_BASE_URL="http://localhost:11434"
OLLAMA_MODEL="llama3.2:3b"
WHISPER_MODEL="base"
WHISPER_LANGUAGE="en"
CONFIDENCE_THRESHOLD=0.70
MAX_AUDIO_SIZE_MB=100
```

### 4. Setup External System Dependencies

#### A. Ollama Model Deployment
Ensure Ollama is running locally and pull the configured model:
```bash
ollama pull llama3.2:3b
ollama list
```

#### B. Local MongoDB Instance
Ensure MongoDB is running locally at the configured URI: `mongodb://localhost:27017`

#### C. System FFmpeg Binaries
Whisper requires FFmpeg for audio processing. Verify it is installed and accessible:
```bash
ffmpeg -version
```

---

## Running the Execution Contexts

### CLI Pipeline Script Execution
Run the pipeline directly against a local audio file. By default, it looks for `clinical_assessment.wav` in the project root:
```bash
python scripts/run_pipeline.py
```

To target a specific file path:
```bash
python scripts/run_pipeline.py path/to/audio.wav
```

The script generates output files in the `output/` directory:
* `output/assessment.json`
* `output/transcript.txt`

### Web Service Runtime
Start the FastAPI development server:
```bash
uvicorn app.main:app --reload
```
Access the interactive Swagger UI documentation at: `http://127.0.0`

---

## API Documentation Map

| HTTP Method | Route | Purpose |
| :--- | :--- | :--- |
| **GET** | `/health` | Returns the service health status. |
| **POST** | `/assessments/parse` | Transcribes WAV, extracts schema, validates confidence, and returns JSON. |
| **POST** | `/assessments` | Saves a previously parsed `FirstAssessment` to MongoDB. |
| **GET** | `/assessments/{assessment_id}` | Retrieves a single assessment by its MongoDB ObjectId. |
| **GET** | `/assessments` | Lists assessments, with optional date filtering. |

### Endpoint Details

#### `POST /assessments/parse`
* Accepts `.wav` files only.
* Enforces file sizes up to `MAX_AUDIO_SIZE_MB`.
* **Flow:** WAV ➔ Whisper ➔ Transcript ➔ LangGraph (Ollama) ➔ FirstAssessment ➔ Confidence Validation ➔ JSON.
* Returns `422` if populated fields fall below the `CONFIDENCE_THRESHOLD`.

#### `POST /assessments`
* Accepts a pre-validated `FirstAssessment` JSON payload.
* Stores the document inside the configured MongoDB collection.

#### `GET /assessments/{assessment_id}`
* Fetches a single document via its hex string ID.
* Returns `404 Not Found` if the ID does not exist.

#### `GET /assessments`
* Supports query parameters: `?from_date=<ISO datetime>` and `?to_date=<ISO datetime>`.
* Returns `422 Unprocessable Entity` if `from_date` is chronologically later than `to_date`.

---

## Test Execution Suite

Execute the automated test suite using pytest:
```bash
pytest -v
```
