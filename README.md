# Stance Health — Voice/Note → Structured Clinical Assessment

A backend pipeline that converts a clinician-patient WAV recording into a structured `FirstAssessment` JSON document.

The system combines local speech-to-text transcription, structured JSON extraction, schema validation, confidence scoring, and MongoDB persistence behind a FastAPI API.

## Architecture

```text
WAV audio
   │
   ▼
Whisper
   │
   │ transcript
   ▼
LangGraph extraction pipeline
   │
   ├── Extract entities
   │      └── Ollama + Qwen2.5 7B
   │
   ├── Validate schema
   │      └── Pydantic v2
   │
   ├── Repair extraction
   │      └── One validation retry
   │
   └── Score confidence
          └── Transcript-grounded lexical matching
   │
   ▼
FirstAssessment
   │
   ▼
FastAPI
   │
   ├── POST /assessments/parse
   ├── POST /assessments
   ├── GET  /assessments/{id}
   └── GET  /assessments
   │
   ▼
MongoDB
```

## Key Features

### Voice-to-structured assessment

A WAV recording is transcribed locally using Whisper. The resulting transcript is passed through a LangGraph extraction pipeline that maps the clinical information into the `FirstAssessment` schema.

### Structured extraction

The extraction model runs locally through Ollama using **Qwen2.5 7B**.

The extraction prompt defines the expected `FirstAssessment` JSON structure and instructs the model to extract only information supported by the transcript.

The resulting JSON is then parsed and validated against the application's Pydantic schema.

### Schema validation

The `FirstAssessment` model uses Pydantic validation with strict field handling. Unexpected fields are rejected before the assessment is accepted by the application.

### Confidence gating

Extracted values are checked against the source transcript using deterministic lexical grounding.

Fields that cannot be sufficiently grounded in the transcript are flagged as low confidence. Assessments that do not meet the configured confidence threshold are returned for review rather than being automatically accepted.

### Validation and repair

If an extraction does not satisfy the expected schema, the validation error is provided to the extraction step and one repair attempt is performed.

The retry is intentionally limited to prevent uncontrolled loops.

### Local processing

The core pipeline runs locally:

* Whisper for speech recognition
* Ollama with Qwen2.5 7B for structured extraction
* MongoDB for persistence

This keeps the development workflow self-contained and avoids requiring an external inference API.

## Project Structure

```text
project/
│
├── app/
│   ├── main.py
│   ├── config.py
│   ├── db.py
│   │
│   ├── models/
│   │   ├── schema.py
│   │   └── db_models.py
│   │
│   ├── routers/
│   │   └── assessments.py
│   │
│   └── services/
│       ├── extraction_agent.py
│       └── transcription.py
│
├── tests/
│   └── test_pipeline.py
│
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

## Requirements

* Python 3.11+
* MongoDB
* Ollama
* Qwen2.5 7B
* Whisper dependencies

Python dependencies are listed in `requirements.txt`.

## Setup

### 1. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

Create the local environment file:

```bash
cp .env.example .env
```

The default configuration uses:

```text
MongoDB: localhost:27017
Ollama: localhost:11434
Whisper model: base
Extraction model: qwen2.5:7b
```

### 4. Start Ollama

Ensure the Qwen2.5 7B model is available:

```bash
ollama pull qwen2.5:7b
```

Start the Ollama service if it is not already running:

```bash
ollama serve
```

### 5. Start MongoDB

For a local MongoDB installation, ensure MongoDB is running on:

```text
mongodb://localhost:27017
```

Alternatively, MongoDB can be started using Docker:

```bash
docker run -d \
  -p 27017:27017 \
  --name stance-mongo \
  mongo:7
```

### 6. Start the API

```bash
uvicorn app.main:app --reload
```

The API will be available at:

```text
http://localhost:8000
```

Interactive API documentation:

```text
http://localhost:8000/docs
```

## API Endpoints

| Method | Endpoint             | Description                                           |
| ------ | -------------------- | ----------------------------------------------------- |
| POST   | `/assessments/parse` | Upload WAV audio and generate a structured assessment |
| POST   | `/assessments`       | Persist a parsed assessment                           |
| GET    | `/assessments/{id}`  | Retrieve a saved assessment                           |
| GET    | `/assessments`       | List saved assessments                                |
| GET    | `/health`            | Check API health                                      |

### Parse an assessment

```bash
curl -X POST \
  -F "file=@path/to/clinical_assessment.wav" \
  http://127.0.0.1:8000/assessments/parse
```

The endpoint returns the structured assessment, transcript, confidence information, and original audio filename.

If the confidence gate is not satisfied, the endpoint returns HTTP `422` together with the fields requiring review.

### Save an assessment

The result from `/assessments/parse` can be submitted to:

```text
POST /assessments
```

The assessment is stored in MongoDB together with its transcript, filename, confidence information, and creation timestamp.

## Example Output

```json
{
  "assessment": {
    "clinicalDetails": {},
    "subjectiveAssessments": [],
    "objectiveAssessment": {
      "tests": []
    },
    "subjectiveGoals": [],
    "objectiveGoals": [],
    "recommendation": [],
    "patientAdvice": {}
  },
  "transcript": "...",
  "overall_confidence": 0.94,
  "low_confidence_fields": [],
  "audio_filename": "clinical_assessment.wav"
}
```

The exact fields are defined by the `FirstAssessment` Pydantic schema.

## Testing

Run the automated tests with:

```bash
pytest tests/ -v
```

The current test suite includes schema-level checks for:

* Valid default assessment construction
* Rejection of unexpected fields

The tests do not require a running LLM, audio model, database, or external API.

The current test run passes both tests:

```text
2 passed
```

## Confidence Configuration

Confidence thresholds can be configured through `.env`:

```text
MIN_FIELD_CONFIDENCE=0.55
MIN_OVERALL_CONFIDENCE=0.6
```

The confidence system is designed as a safety gate between extraction and persistence.

## Configuration

The main environment variables are:

```text
MONGO_URI=mongodb://localhost:27017
MONGO_DB_NAME=stance_health
MONGO_COLLECTION=assessments

WHISPER_MODEL=base
WHISPER_DEVICE=cpu

OLLAMA_BASE_URL=http://localhost:11434
EXTRACTION_MODEL=qwen2.5:7b

MIN_FIELD_CONFIDENCE=0.55
MIN_OVERALL_CONFIDENCE=0.6

UPLOAD_DIR=./data/uploads
```

## Design Considerations

### Transcript-grounded extraction

The extraction result is checked against the original transcript instead of relying only on the extraction model's confidence.

This provides a deterministic validation layer that can identify unsupported values before persistence.

### Separation of schemas

The frontend-facing `FirstAssessment` schema is kept separate from the MongoDB storage document.

The storage document adds persistence metadata such as:

* MongoDB `_id`
* creation timestamp
* audio filename
* confidence metadata

This avoids coupling storage concerns to the clinical assessment schema.

### Limited repair loop

The extraction process allows one schema-repair attempt. If the result remains invalid after the retry, the pipeline fails rather than repeatedly attempting extraction.

## Known Limitations

* Lexical confidence scoring can flag valid paraphrases because the extracted wording may differ from the transcript.
* The current API does not include authentication or multi-tenant access control.
* Whisper model size can be increased for potentially better transcription quality at the cost of additional compute and latency.
* The current automated tests focus on schema behavior rather than full end-to-end audio and database integration.

## Future Improvements

Potential improvements include:

* Semantic entailment-based confidence scoring
* Authentication and authorization
* Multi-tenant data isolation
* Integration tests covering the complete audio-to-database flow
* Improved clinical terminology normalization
* Configurable Whisper model selection
* Production monitoring and structured logging
