# Clinical Audio to FirstAssessment JSON Report Pipeline

A Python service that transcribes clinical audio recordings (`.wav`), extracts structured clinical data into the exact `FirstAssessment` Pydantic schema using a **LangGraph agent** powered by **Groq** (`whisper-large-v3` + `openai/gpt-oss-120b`), and stores/retrieves assessment reports in **MongoDB**.

---

## 🚀 Setup Instructions

### 1. Prerequisites
* Python 3.10+
* Virtual Environment (`venv`)
* Groq API Key (used for Whisper audio transcription and LangGraph LLM extraction)
* MongoDB (local instance `mongodb://localhost:27017` or cloud MongoDB URI)

### 2. Environment Configuration

Create a `.env` file in the root directory:

```ini
GROQ_API_KEY=your_groq_api_key_here
GROQ_WHISPER_MODEL=whisper-large-v3
GROQ_LLM_MODEL=openai/gpt-oss-120b

# Optional: MongoDB Configuration (Defaults to local)
MONGODB_URI=mongodb://localhost:27017
DB_NAME=stance_health
```

### 3. Installation

Activate your virtual environment and install project dependencies:

```bash
# Windows
.\venv\Scripts\activate

# Linux/macOS
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

### 4. Running the Test Pipeline Script

Place your clinical audio WAV file (e.g. `clinical_assessment.wav`) in the project root directory (or provide the path to your file), then execute the pipeline test script:

```bash
python test_pipeline.py clinical_assessment.wav
```

You can also specify a path to any custom audio file:
```bash
python test_pipeline.py path/to/your_audio_file.wav
```

### 5. Running the FastAPI Web Service

Start the FastAPI application with Uvicorn:

```bash
uvicorn app.main:app --reload --port 8000
```

Access interactive API documentation at: `http://127.0.0.1:8000/docs`

---

## 💡 Key Design Decisions

1. **Strict Schema Integrity & Type Coercion (`app/schemas`)**:
   * Applied reusable Pydantic v2 validators (`EmptyStr`) across all 7 sections to convert `null` inputs into empty strings (`""`), strictly satisfying the constraint *"All string fields must be strings, not null."*
   * Coerced `null` section arrays into empty lists (`[]`) and single objects into single-element lists, guaranteeing *"All array fields must be arrays even if only one item is present."*

2. **Decoupled Architecture by Request Path (`app/`)**:
   * `routes/`: Handles HTTP requests and responses exclusively.
   * `services/`: Encapsulates audio transcription, LangGraph agent execution, and MongoDB persistence.
   * `schemas/`: Defines the single source of truth for input/output data contracts.
   * `config.py`: Reads environment variables once at startup.

3. **High-Accuracy Transcription biased for Clinical Terms (`app/services/transcription.py`)**:
   * Selected Groq's hosted `whisper-large-v3` with a specialized clinical prompt (`CLINICAL_TRANSCRIPTION_PROMPT`) containing medical terminology (e.g., *ROM*, *goniometer*, *dorsiflexion*, *flexion*, *extension*).
   * Spoken negative clinical degrees (e.g. *"negative 5 degrees"*) are transcribed and extracted as clean numeric values (`"-5"`), preventing raw phonetic artifacts from entering production JSON.

4. **Structured LangGraph Extraction Agent & Confidence Gate (`app/services/agent.py`)**:
   * Built using a compiled LangGraph `StateGraph` with node-based execution (`extract` $\rightarrow$ `validate`).
   * The `validate` node serves as the sole gate: if the extraction confidence score falls below `0.70`, `POST /assessments/parse` raises **HTTP 422 Unprocessable Entity** with field-level error details (`field_errors`).

5. **Asynchronous Database Access (`app/services/storage.py`)**:
   * Used Motor for non-blocking MongoDB interactions across all persistence endpoints (`POST /assessments`, `GET /assessments/{id}`, `GET /assessments`).
