# Voice/Note → Structured Clinical Assessment Form Filler

An end-to-end clinical assessment pipeline that converts a clinician's voice recording into a structured, validated, grounded clinical assessment.

The system performs:

WAV Audio → Local Whisper → Clinical Transcript → LangGraph Extraction → Pydantic Validation → Grounding Verification → Confidence Scoring → Structured JSON → FastAPI + MongoDB

---

## Overview

This project implements a production-oriented pipeline for transforming unstructured clinical voice notes into a structured FirstAssessment object.

The pipeline is designed around four important principles:

1. The transcript is the source of truth
2. The LLM must produce structured data
3. Unsupported information must not be invented
4. Uncertain or missing information must remain explicitly uncertain

The system separates transcription, extraction, validation, grounding, persistence, and API concerns so each component can be tested and replaced independently.

---

## Architecture

                    ┌──────────────────────┐
                    │   WAV / Audio Input  │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │      Audio I/O       │
                    │ Validation / Loading │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   Local Whisper STT  │
                    │  Speech → Text       │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │  Clinical Transcript │
                    └──────────┬───────────┘
                               │
                               ▼
              ┌─────────────────────────────────┐
              │           LangGraph             │
              │                                 │
              │  ┌───────────────────────────┐  │
              │  │ Clinical Extraction       │  │
              │  ├───────────────────────────┤  │
              │  │ Subjective Extraction     │  │
              │  ├───────────────────────────┤  │
              │  │ Objective Extraction      │  │
              │  ├───────────────────────────┤  │
              │  │ Goals Extraction           │  │
              │  ├───────────────────────────┤  │
              │  │ Plan Extraction            │  │
              │  └───────────────────────────┘  │
              └───────────────┬─────────────────┘
                              │
                              ▼
                    ┌──────────────────────┐
                    │ Pydantic Validation  │
                    │   FirstAssessment    │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Grounding Verification│
                    │ Transcript Evidence  │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Confidence Scoring   │
                    │ + Uncertainty Flags  │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   Structured JSON    │
                    └──────────┬───────────┘
                               │
                    ┌──────────┴───────────┐
                    ▼                      ▼
          ┌──────────────────┐   ┌──────────────────┐
          │     FastAPI      │   │     MongoDB      │
          │    REST API      │   │    Persistence   │
          └──────────────────┘   └──────────────────┘

---

## Features

* WAV/audio validation and preprocessing
* Local Whisper speech-to-text
* Clinical transcript generation
* LangGraph-based extraction workflow
* Configurable LLM provider
* Local Ollama LLM support
* Provider abstraction for optional hosted/API LLMs
* Clinical information extraction
* Subjective extraction
* Objective extraction
* Goals extraction
* Plan extraction
* Strict Pydantic FirstAssessment schema
* Structured JSON output
* Transcript-based grounding verification
* Unsupported-information detection
* Missing-information detection
* Uncertainty handling
* Confidence scoring
* Field-level confidence information
* FastAPI REST API
* Automatic OpenAPI/Swagger documentation
* MongoDB persistence
* Repository pattern for database access
* CLI transcription utility
* CLI complete pipeline execution
* Environment-based configuration
* Secure secret handling through environment variables
* Unit tests
* Integration tests
* End-to-end pipeline tests
* API tests
* Repository tests
* Audio tests
* Grounding tests
* Confidence tests
* Schema contract tests

---

## 1. Project Structure

assignments/
│
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── config.py
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes.py
│   │   └── schemas.py
│   │
│   ├── schemas/
│   │   ├── __init__.py
│   │   └── assessment.py
│   │
│   ├── transcription/
│   │   ├── __init__.py
│   │   ├── audio_io.py
│   │   └── whisper_service.py
│   │
│   ├── extraction/
│   │   ├── __init__.py
│   │   ├── graph.py
│   │   ├── llm.py
│   │   ├── prompts.py
│   │   ├── grounding.py
│   │   └── confidence.py
│   │
│   ├── db/
│   │   ├── __init__.py
│   │   ├── client.py
│   │   ├── models.py
│   │   └── repository.py
│   │
│   └── services/
│       ├── __init__.py
│       └── pipeline.py
│
├── scripts/
│   ├── run_pipeline.py
│   └── transcribe.py
│
├── tests/
│   ├── conftest.py
│   ├── test_schema_contract.py
│   ├── test_audio_io.py
│   ├── test_extraction_graph.py
│   ├── test_grounding.py
│   ├── test_confidence.py
│   ├── test_repository.py
│   ├── test_api.py
│   └── test_pipeline.py
│
├── data/
│   ├── clinical_assessment.wav
│   └── sample_output.json
│
├── docs/
│   ├── sample-report.pdf
│   └── screenshots/
│
├── .env
├── .env.example
├── .gitignore
├── requirements.txt
├── requirements-optional.txt
├── pyproject.toml
└── README.md

---

## 2. Technology Stack

Component           | Technology
-------------------|-----------------------------------
Language           | Python
API                | FastAPI
Validation         | Pydantic
Speech-to-Text     | Whisper
Graph Workflow     | LangGraph
LLM                | Configurable provider
Local LLM          | Ollama
Database           | MongoDB
Database Driver    | PyMongo
Testing            | Pytest
API Documentation  | OpenAPI / Swagger
Configuration      | Environment variables

---

## 3. Speech-to-Text

The transcription layer uses Whisper for local speech recognition.

Audio
  ↓
Audio validation
  ↓
Whisper
  ↓
Clinical transcript

The transcription implementation is isolated behind:

app/transcription/whisper_service.py

This allows the speech-to-text implementation to be changed without modifying the extraction or API layers.

Example:

transcript = whisper_service.transcribe(
    "data/clinical_assessment.wav"
)

The resulting transcript becomes the primary evidence source for downstream extraction and grounding.

---

## 4. LLM Extraction

The extraction system uses a configurable LLM abstraction.
The default architecture supports a local LLM through Ollama.

Transcript
    ↓
Prompt
    ↓
LLM
    ↓
Structured extraction

The LLM provider is isolated behind:

app/extraction/llm.py

This prevents the rest of the application from being tightly coupled to one model provider.

The system can therefore support:
* Local Ollama models
* Optional hosted/API providers
* Future LLM providers without changing the pipeline architecture

The application should never hard-code API keys or secrets.
All provider configuration is loaded from environment variables.

---

## 5. LangGraph Extraction Pipeline

The extraction workflow is implemented using LangGraph.
The graph separates the extraction responsibilities into logical stages.

Transcript
    ↓
Clinical extraction
    ↓
Subjective extraction
    ↓
Objective extraction
    ↓
Goals extraction
    ↓
Plan extraction
    ↓
Structured assessment

The graph implementation is located at:

app/extraction/graph.py

Prompts are maintained separately in:

app/extraction/prompts.py

This makes the prompts easier to test, maintain, and improve.

---

## 6. Clinical Assessment Schema

The extracted information is validated using a strict Pydantic model.
The canonical schema is located at:

app/schemas/assessment.py

The main model is:

FirstAssessment

The schema acts as the contract between the extraction layer and the rest of the application.
The LLM output is never trusted directly.

Instead:

LLM output
    ↓
Pydantic validation
    ↓
FirstAssessment
    ↓
Grounding
    ↓
Confidence
    ↓
Final JSON

This prevents malformed LLM output from silently entering the system.

---

## 7. Grounding Verification

LLMs can produce information that was not present in the transcript.
To reduce hallucination, every extracted field is checked against the original transcript.

Extracted field
      ↓
Search for transcript evidence
      ↓
Supported?
   /       \
 YES       NO
  ↓         ↓
Accept    Flag / reject

The grounding implementation is located at:

app/extraction/grounding.py

The transcript is treated as the source of truth.
The system should not invent:
* Diagnoses
* Symptoms
* Measurements
* Goals
* Treatment plans
* Medications
* Clinical observations
* Patient history

when the information is not supported by the transcript.

---

## 8. Confidence Scoring

Extracted information receives confidence information based on factors such as:
* Transcript evidence
* Extraction certainty
* Grounding result
* Missing information
* Explicit uncertainty
* Unsupported information

The implementation is located at:

app/extraction/confidence.py

The system explicitly distinguishes between:
* Known
* Unknown
* Uncertain
* Unsupported

This is important because absence of information should not be converted into a fabricated value.

---

## 9. End-to-End Pipeline

The complete service is implemented in:

app/services/pipeline.py

The complete flow is:

Audio
 ↓
Audio validation
 ↓
Whisper transcription
 ↓
Clinical transcript
 ↓
LangGraph extraction
 ↓
Pydantic FirstAssessment
 ↓
Grounding verification
 ↓
Confidence scoring
 ↓
Final structured JSON
 ↓
MongoDB persistence

The pipeline service is responsible for coordinating the individual components without placing all business logic inside the API routes.

---

## 10. FastAPI

The application exposes a REST API using FastAPI.
API implementation:

app/api/routes.py

Application entry point:

app/main.py

Run the API with:

uvicorn app.main:app --reload

The API documentation is automatically available through FastAPI's OpenAPI integration.

Swagger UI: /docs
ReDoc: /redoc

---

## 11. API Capabilities

The API is designed around the complete assessment lifecycle.
Typical operations include:

POST   /assessments
GET    /assessments/{id}
GET    /assessments
PUT    /assessments/{id}

The exact request and response schemas are defined in:

app/api/schemas.py

The API layer is responsible for:
* Request validation
* Response validation
* Error handling
* Pipeline invocation
* Persistence
* Retrieval

The extraction logic remains outside the API layer.

---

## 12. MongoDB Persistence

MongoDB is used for assessment persistence.
Database implementation:

app/db/client.py
app/db/models.py
app/db/repository.py

The architecture follows a repository pattern:

FastAPI
   ↓
Service
   ↓
Repository
   ↓
MongoDB

This prevents database-specific code from spreading throughout the application.
The repository layer handles operations such as:
* Create assessment
* Get assessment
* List assessments
* Update assessment
* Delete assessment
* Filter/query assessments

MongoDB configuration is provided through environment variables.

---

## 13. Configuration

Configuration is centralized in:

app/config.py

Environment-specific values should be stored in:

.env

An example configuration is provided in:

.env.example

Example:

APP_ENV=development

MONGODB_URI=mongodb://localhost:27017
MONGODB_DATABASE=clinical_assessment

LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1

WHISPER_MODEL=base

Do not commit real secrets to Git.

---

## 14. Local LLM with Ollama

The application supports running the LLM locally through Ollama.

Start Ollama:

ollama serve

Pull a supported model:

ollama pull llama3.1

Then configure:

LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1

The LLM implementation remains isolated from the rest of the application.
This makes the system suitable for development environments where clinical data should remain local.
Hosted/API LLM providers can be added through the same provider abstraction without redesigning the pipeline.

---

## 15. Whisper Setup

Install the project dependencies:

pip install -r requirements.txt

The application uses Whisper for speech recognition.
The Whisper model can be configured through the environment:

WHISPER_MODEL=base

Available model sizes depend on the installed Whisper implementation and available hardware.
For development, a smaller model can be used for faster execution.
For higher transcription quality, a larger model can be configured when sufficient compute is available.

---

## 16. Running the Complete Pipeline

The complete pipeline can be executed using:

python scripts/run_pipeline.py

With the sample audio:

data/clinical_assessment.wav

The process performs:

1. Load audio
2. Validate audio
3. Transcribe with Whisper
4. Extract clinical information
5. Validate FirstAssessment
6. Ground extracted information
7. Calculate confidence
8. Generate structured JSON
9. Persist assessment

---

## 17. Transcription CLI

Transcription can also be executed independently.

python scripts/transcribe.py data/clinical_assessment.wav

This is useful for debugging the speech-to-text stage separately from the LLM pipeline.

---

## 18. Example Output

A simplified example of the final structure:

{
  "subjective": {
    "chief_complaint": null,
    "history": null,
    "symptoms": []
  },
  "objective": {
    "observations": [],
    "measurements": []
  },
  "goals": [],
  "plan": [],
  "confidence": {},
  "grounding": {}
}

The exact structure must follow the canonical FirstAssessment Pydantic contract implemented in:

app/schemas/assessment.py

Missing information should remain missing rather than being guessed.

---

## 19. Error Handling

The application explicitly handles failures at each stage.
Examples include:
* Invalid audio
* Missing audio file
* Unsupported audio format
* Whisper failure
* LLM unavailable
* Invalid LLM output
* Pydantic validation failure
* Grounding failure
* MongoDB unavailable
* Invalid API request
* Assessment not found

Errors should be converted into clear application-level responses rather than exposing raw internal exceptions.

---

## 20. Testing

Tests are separated by responsibility.

tests/
├── conftest.py
├── test_schema_contract.py
├── test_audio_io.py
├── test_extraction_graph.py
├── test_grounding.py
├── test_confidence.py
├── test_repository.py
├── test_api.py
└── test_pipeline.py

Run all tests:

pytest -v

Run a specific test:

pytest tests/test_grounding.py -v

Run the complete suite:

pytest tests/ -v

The test suite covers:
* Schema validation
* Audio handling
* Extraction graph
* Grounding
* Confidence scoring
* Database repository
* API behavior
* Complete pipeline integration

---

## 21. Testing Philosophy

The system should be tested at multiple levels.

Unit tests:
Individual components are tested independently (audio_io, grounding, confidence, schemas, repository).

Integration tests:
Multiple components are tested together (LLM → extraction → schema, repository → MongoDB).

End-to-end tests:
The complete pipeline is tested (Audio → Whisper → Extraction → Validation → Grounding → Confidence → Persistence).

External services should be mocked where appropriate so the test suite remains deterministic.

---

## 22. Separation of Concerns

The project intentionally separates responsibilities.

api/
    HTTP and REST concerns
schemas/
    Data contracts
transcription/
    Speech recognition
extraction/
    LLM and clinical extraction
db/
    MongoDB persistence
services/
    Application orchestration
tests/
    Verification

This makes the application easier to maintain, debug, test, and extend.

---

## 23. Security and Privacy

Clinical information can be sensitive.
The application therefore follows these principles:
* Do not hard-code secrets
* Do not commit .env
* Use .env.example for configuration examples
* Keep provider credentials outside source code
* Avoid logging sensitive clinical content unnecessarily
* Treat the transcript as sensitive data
* Avoid exposing internal exceptions through the API
* Validate incoming files
* Validate structured outputs
* Prevent unsupported LLM-generated information from being accepted automatically

For local deployments, Whisper and Ollama can run locally so audio and transcript data do not need to leave the development environment.

---

## 24. Design Principles

The implementation follows these principles:

* Transcript-first: The transcript is the primary evidence source.
* Structured-first: LLM output is converted into a strict Pydantic model.
* Evidence-based: Extracted values should be supported by transcript evidence.
* Uncertainty-aware: The system does not turn unknown information into fabricated facts.
* Provider-independent: LLM and transcription providers are isolated behind service interfaces.
* Testable: Every important component can be tested independently.
* Replaceable: Whisper, Ollama, MongoDB, and other infrastructure components can be replaced without rewriting the entire application.

---

## 25. Complete Data Flow

                 AUDIO
                   │
                   ▼
           ┌───────────────┐
           │   Audio I/O   │
           │ Validate file │
           └───────┬───────┘
                   │
                   ▼
           ┌───────────────┐
           │    Whisper    │
           │ Speech → Text │
           └───────┬───────┘
                   │
                   ▼
              TRANSCRIPT
                   │
                   ▼
           ┌───────────────┐
           │   LangGraph   │
           └───────┬───────┘
                   │
        ┌──────────┼──────────┐
        ▼          ▼          ▼
   Subjective  Objective    Goals
        │          │          │
        └──────────┼──────────┘
                   │
                   ▼
                 Plan
                   │
                   ▼
          ┌────────────────┐
          │ FirstAssessment│
          │   Pydantic     │
          └───────┬────────┘
                  │
                  ▼
          ┌────────────────┐
          │    Grounding   │
          │ Transcript     │
          │ Verification   │
          └───────┬────────┘
                  │
                  ▼
          ┌────────────────┐
          │   Confidence   │
          │    Scoring     │
          └───────┬────────┘
                  │
                  ▼
            STRUCTURED JSON
                  │
           ┌──────┴──────┐
           ▼             ▼
       FastAPI        MongoDB

---

## 26. Example Development Workflow

Clone the repository:

git clone <repository-url>
cd assignments

Create a virtual environment:

python -m venv .venv

Activate it on macOS/Linux:

source .venv/bin/activate

Install dependencies:

pip install -r requirements.txt

Create environment configuration:

cp .env.example .env

Start MongoDB.
Start Ollama if using the local provider:

ollama serve

Start the API:

uvicorn app.main:app --reload

Open:

http://127.0.0.1:8000/docs

Run the pipeline:

python scripts/run_pipeline.py

Run tests:

pytest -v

---

## 27. Production-Oriented Architecture

The project is intentionally structured so that it can evolve beyond a coding assignment.
A future production deployment can separate:

Frontend
    ↓
API Gateway
    ↓
FastAPI
    ↓
Pipeline Service
    ├── Transcription Service
    ├── Extraction Service
    ├── Grounding Service
    └── Confidence Service
            ↓
        MongoDB

Additional infrastructure such as queues, authentication, observability, caching, and background workers can be introduced without changing the core domain models.

---

## 28. Extensibility

The architecture can later support:
* Additional audio formats
* Additional speech-to-text providers
* Additional LLM providers
* Multiple local Ollama models
* Hosted LLM APIs
* Authentication
* Role-based access
* Background processing
* Job queues
* Audit logs
* Assessment versioning
* Human review workflows
* Field-level provenance
* Better clinical terminology normalization
* Additional assessment types
* Frontend clinical forms
* Export to PDF
* FHIR-compatible integrations

These are extension points rather than requirements for the core pipeline.

---

## 29. Key Implementation Contract

The central contract is:

Audio
  ↓
Transcript
  ↓
Evidence-based extraction
  ↓
FirstAssessment
  ↓
Grounding
  ↓
Confidence
  ↓
Persisted / API-ready JSON

At no point should unsupported clinical information be silently invented.
If the transcript does not provide enough evidence for a field, the system should represent the information as missing or uncertain according to the FirstAssessment contract.

---

## 30. Final Goal

The final application provides a complete voice-to-structured-assessment workflow:

Clinician Voice Note
        ↓
     Whisper
        ↓
    Transcript
        ↓
    LangGraph
        ↓
   LLM Extraction
        ↓
 Pydantic Validation
        ↓
Grounding Verification
        ↓
Confidence Scoring
        ↓
 Structured Assessment
        ↓
 ┌───────────────┐
 │    FastAPI    │
 │      +        │
 │    MongoDB    │
 └───────────────┘

The result is a modular, testable, configurable, and evidence-grounded clinical assessment pipeline suitable for the assignment and designed with production-oriented separation of concerns.