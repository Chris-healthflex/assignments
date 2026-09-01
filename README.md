# Clinical Assessment API

A FastAPI-based clinical assessment pipeline that converts clinician–patient WAV recordings into structured `FirstAssessment` data using Whisper, Groq/LangChain extraction, Pydantic validation, LangGraph, and MongoDB.

## Features

* WAV audio upload
* Local Whisper transcription
* Clinical information extraction using Groq
* Structured `FirstAssessment` validation with Pydantic v2
* Confidence-based extraction validation
* HTTP `422` response for low-confidence extracted fields
* LangGraph extraction workflow
* MongoDB persistence
* Retrieve assessment by ID
* List all assessments
* Filter assessments by date
* Temporary audio-file cleanup
* Input and error validation

## Tech Stack

* Python 3.10+
* FastAPI
* Uvicorn
* OpenAI Whisper
* Groq API
* LangGraph
* Pydantic v2
* MongoDB / PyMongo

## Project Structure

```text
project/
│
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── extractor.py
│   ├── graph.py
│   ├── database.py
│   ├── schemas.py
│   └── whisper_service.py
│
├── test_extraction.py
├── requirements.txt
├── .env
└── README.md
```

## Setup

### 1. Create virtual environment

```bash
python -m venv .venv
```

### 2. Activate it

Windows:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file:

```env
GROQ_API_KEY=your_groq_api_key
MONGODB_URI=your_mongodb_connection_string
MONGODB_DATABASE=clinical_assessment
```

Do **not** commit `.env` or API keys to GitHub.

## Running the API

```bash
uvicorn app.main:app --reload
```

API:

```text
http://127.0.0.1:8000
```

Swagger documentation:

```text
http://127.0.0.1:8000/docs
```

## API Endpoints

### 1. Health Check

```http
GET /health
```

Returns:

```json
{
  "status": "healthy"
}
```

### 2. Parse Clinical Assessment

```http
POST /assessments/parse
```

Upload a `.wav` file.

Pipeline:

```text
WAV
 ↓
Whisper
 ↓
Transcription
 ↓
Groq Clinical Extraction
 ↓
Pydantic Validation
 ↓
FirstAssessment
```

The parsing endpoint does **not** save the assessment to MongoDB.

### 3. Save Assessment

```http
POST /assessments
```

Accepts a validated `FirstAssessment` JSON object and stores it in MongoDB.

### 4. Get Assessment

```http
GET /assessments/{assessment_id}
```

Retrieves a previously stored assessment using its MongoDB ObjectId.

### 5. List Assessments

```http
GET /assessments
```

Returns saved assessments ordered by creation time.

### Date Filtering

```http
GET /assessments?date=2026-09-01
```

Date format:

```text
YYYY-MM-DD
```

## Low-Confidence Handling

The extraction pipeline validates confidence before accepting extracted clinical fields.

If one or more required extracted fields are below the configured confidence threshold, the API returns:

```text
HTTP 422 Unprocessable Entity
```

with field-level error information instead of silently accepting unreliable clinical data.

This helps prevent unsupported or uncertain clinical values from being treated as confirmed information.

## Clinical Safety

The extraction process is designed to:

* Extract only information supported by the transcription
* Avoid inventing clinical measurements
* Avoid hallucinating diagnoses, scores, dates, or values
* Preserve unavailable information as empty rather than fabricating it
* Validate extracted output against the `FirstAssessment` Pydantic schema
* Reject low-confidence extraction

## LangGraph Workflow

The LangGraph workflow contains the clinical extraction node:

```text
Transcription
      ↓
Clinical Extraction
      ↓
FirstAssessment
      ↓
MongoDB
```

The graph is compiled as `clinical_graph` and can be invoked from the extraction test workflow.

## Testing

Run:

```bash
python test_extraction.py
```

The test:

1. Loads the Whisper model.
2. Transcribes the WAV recording.
3. Sends the transcription through the LangGraph extraction pipeline.
4. Produces a structured `FirstAssessment`.
5. Displays the extracted clinical assessment.

## Example Output

```json
{
  "clinicalDetails": {
    "clinicalHistory": "...",
    "chiefComplaint": "...",
    "duration": "..."
  },
  "subjectiveAssessments": [],
  "objectiveAssessment": {
    "tests": []
  },
  "subjectiveGoals": [],
  "objectiveGoals": [],
  "recommendation": [],
  "patientAdvice": {
    "adviceDetails": ""
  }
}
```

## Error Handling

The API handles:

* Missing audio file
* Empty WAV files
* Non-WAV uploads
* Failed transcription
* Invalid extraction output
* Pydantic validation errors
* Low-confidence extraction
* Invalid MongoDB ObjectIds
* Missing assessments
* Invalid date filters
* MongoDB/API errors

## Important

Add the following to `.gitignore`:

```gitignore
.venv/
__pycache__/
.env
*.pyc
```

Never commit:

```text
.env
GROQ_API_KEY
MongoDB credentials
```

## Quick Start

```bash
git clone <repository-url>
cd <project-folder>

python -m venv .venv
.venv\Scripts\activate

pip install -r requirements.txt

# Configure .env

uvicorn app.main:app --reload
```

Then open:

```text
http://127.0.0.1:8000/docs
```

and test the four assessment endpoints directly through Swagger UI.
