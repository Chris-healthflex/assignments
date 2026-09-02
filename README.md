# Stance Health Clinical Assessment API

A FastAPI pipeline that converts a physiotherapy clinical WAV recording into the structured `FirstAssessment` JSON format required by the Stance Health frontend.

## Tech Stack

*   **Python 3.10**
*   **FastAPI**
*   **OpenAI Whisper**
*   **LangGraph**
*   **Ollama**
*   **Pydantic v2**
*   **MongoDB**
*   **Pytest**

## Pipeline Flow

```text
[WAV Audio] -> [Whisper Transcription] -> [LangGraph + Ollama Extraction] -> [Pydantic Validation] -> [FirstAssessment JSON] -> [MongoDB]
```

## Project Structure

```text
stance-assessment-app/
├── app/
│   ├── models/
│   ├── repositories/
│   ├── services/
│   ├── config.py
│   ├── database.py
│   └── main.py
├── scripts/
│   └── run_pipeline.py
├── tests/
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

## Setup

### 1. Create Virtual Environment
```bash
python -m venv venv
# Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# Linux/macOS:
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure Environment
Create a `.env` file in the project root:

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

### 4. External Dependencies

#### Ollama
Download and verify the local model (the project uses Ollama locally and does not require a paid LLM API):
```bash
ollama pull llama3.2:3b
ollama list
```

#### MongoDB
The default configuration uses a local MongoDB instance (`mongodb://localhost:27017`) and the database `stance_assessment`. MongoDB Atlas can also be used by changing the `MONGODB_URI` in your `.env` file.

#### FFmpeg
Whisper requires FFmpeg installed on your system. Verify your installation with:
```bash
ffmpeg -version
```

---

## Running the Pipeline (CLI)

1. Place your clinical audio file in the project root named `clinical_assessment.wav`.
2. Run the script:
   ```bash
   python scripts/run_pipeline.py
   ```

**What the pipeline does:**
*   Transcribes the WAV file using Whisper.
*   Extracts clinical information using LangGraph and Ollama.
*   Validates the result using Pydantic.
*   Saves the structured assessment and transcript.

**Generated files:**
*   `output_assessment.json`
*   `transcript.txt`

---

## Running the API

Start the FastAPI development server:
```bash
uvicorn app.main:app --reload
```

*   **API Base URL:** `http://127.0.0.1:8000`
*   **Swagger Documentation:** `http://127.0.0`

### API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| **GET** | `/health` | Health check |
| **POST** | `/assessments/parse` | Upload WAV and extract assessment |
| **POST** | `/assessments` | Save assessment to MongoDB |
| **GET** | `/assessments/{id}` | Retrieve specific assessment |
| **GET** | `/assessments` | List assessments (supports optional `from_date` and `to_date` filters) |

---

## Testing

Run the test suite using pytest:
```bash
pytest -q
```