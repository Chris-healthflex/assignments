# Clinical Assessment Extraction Pipeline

## Project Overview
This project is a FastAPI-based backend service designed to process clinical assessment audio recordings (WAV format), transcribe them using OpenAI Whisper, and extract highly structured clinical information into a precise JSON format (using Gemini through LangGraph/LangChain). The structured assessments are stored in a MongoDB database and made available through a RESTful API.

## Architecture
- **Transcription**: OpenAI Whisper (local model `base`)
- **LLM / Extraction**: Google Gemini (gemini-2.5-flash-lite) driven by LangChain & LangGraph
- **Data Validation**: Pydantic v2 ensures strict schema adherence (no extra fields, no null strings).
- **Storage**: MongoDB
- **API**: FastAPI

## Requirements
- Python 3.10+
- FFmpeg (required for Whisper)
- MongoDB instance (local or Atlas)
- Google Gemini API Key

## Installation
1. Clone the repository and navigate to the root directory.
2. Create and activate a Python virtual environment:
   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # macOS/Linux
   source .venv/bin/activate
   ```
3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Environment Variables
Create a `.env` file in the root directory with the following variables:
```env
MONGODB_URI=mongodb://localhost:27017/
MONGODB_DATABASE=clinical_assessment_db
MONGODB_COLLECTION=assessments
GOOGLE_API_KEY=your-gemini-api-key
```

## MongoDB Setup
The application expects a running MongoDB instance. By default, it connects to a local instance at `mongodb://localhost:27017/`. You can update `MONGODB_URI` in `.env` to connect to a different database instance or MongoDB Atlas.

## How to run FastAPI
Start the server using uvicorn:
```bash
uvicorn app.main:app --reload
```
The API documentation will be accessible at http://127.0.0.1:8000/docs.

## How to run tests
To execute the automated API tests:
```bash
python -m pytest -q tests/
```

## How to run the pipeline
A test script is provided to verify the end-to-end pipeline with the included `data/clinical_assessment.wav` file.
```bash
python test_pipeline.py
```
This script runs the file through the pipeline without saving to the database unless all extractions pass the confidence threshold.

## API Endpoints
- `GET /health` : Health check endpoint.
- `POST /assessments/parse` : Transcribe a WAV file and return a valid FirstAssessment JSON (does not save to DB).
- `POST /assessments` : Save a valid FirstAssessment to MongoDB.
- `GET /assessments` : List saved assessments (supports `date_from` and `date_to` filters).
- `GET /assessments/{id}` : Get an assessment by its MongoDB ObjectId.

## Example API usage
**Parse an audio file (returns 422 if confidence is low):**
```bash
curl -X POST "http://127.0.0.1:8000/assessments/parse" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@data/clinical_assessment.wav"
```

## Confidence/error behavior
The system strictly prohibits hallucination and enforces a confidence threshold of 0.70. If the extraction model encounters ambiguous terminology or cannot confidently extract a value (e.g., "negic 5"), the field is flagged and the `/assessments/parse` endpoint returns an HTTP 422 Unprocessable Entity with detailed field-level error messages.

## Known Limitations
- The system treats transcription errors conservatively; it prefers to reject ambiguous phrases rather than guessing clinical intent.
- Whisper `base` model may struggle with heavy accents or highly specialized medical jargon, increasing the likelihood of 422 responses.
