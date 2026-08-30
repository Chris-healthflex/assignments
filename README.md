# Stance Health Clinical Assessment Pipeline

Python backend for parsing a clinician-patient WAV recording into a validated `FirstAssessment`.

The service accepts audio, transcribes it with local Whisper, extracts structured assessment data with a small LangGraph flow, validates the result with Pydantic v2, and stores saved assessments in MongoDB.

## Setup

Use Python 3.10 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Whisper also needs `ffmpeg` available on your machine:

```bash
brew install ffmpeg
```

The first local Whisper run downloads the configured model.

## Environment

```env
MONGODB_URI=mongodb://localhost:27017
MONGODB_DATABASE=stance
MONGODB_COLLECTION=assessments
WHISPER_MODEL=base
OPENAI_API_KEY=
CONFIDENCE_THRESHOLD=0.6
```

`OPENAI_API_KEY` is required for the LangChain/OpenAI extraction step. Tests mock that path and do not call OpenAI.

## MongoDB

Run MongoDB locally with Docker:

```bash
docker run --rm -p 27017:27017 --name stance-mongo mongo:7
```

The app uses the `stance.assessments` collection by default.

## Run The API

```bash
uvicorn app.main:app --reload
```

Open the interactive docs at:

```text
http://127.0.0.1:8000/docs
```

## API Examples

Parse a WAV without saving it:

```bash
curl -X POST http://127.0.0.1:8000/assessments/parse \
  -F "file=@data/clinical_assessment.wav;type=audio/wav"
```

Save a validated assessment:

```bash
curl -X POST http://127.0.0.1:8000/assessments \
  -H "Content-Type: application/json" \
  -d '{
    "clinicalDetails": {
      "clinicalHistory": "",
      "chiefComplaint": "right knee pain",
      "duration": "two weeks"
    },
    "subjectiveAssessments": [],
    "objectiveAssessment": { "tests": [] },
    "subjectiveGoals": [],
    "objectiveGoals": [],
    "recommendation": [],
    "patientAdvice": { "adviceDetails": "" }
  }'
```

Fetch one assessment:

```bash
curl http://127.0.0.1:8000/assessments/<id>
```

List assessments, optionally filtered by UTC creation date:

```bash
curl http://127.0.0.1:8000/assessments
curl "http://127.0.0.1:8000/assessments?date=2026-08-30"
```

## Tests

```bash
pytest
```

The test suite does not require MongoDB, Whisper, or OpenAI. External services are mocked or bypassed where appropriate.

If `data/clinical_assessment.wav` exists, the fixture test confirms it is available. The repository does not include a sample recording.

## Design Decisions

The Pydantic models are the final authority for the response shape. Every model forbids extra fields, strings default to `""`, and lists default to `[]`.

The extraction prompt tells the model to use only transcript evidence. It explicitly rejects guessing numbers, dates, laterality, diagnoses, treatment frequency, and goals. If something is unclear or contradicted, the field stays empty.

Confidence scores are internal. The API returns only the validated `FirstAssessment` from `/assessments/parse`. If confidence falls below `CONFIDENCE_THRESHOLD`, the API returns `422` with the affected fields or sections.

The LangGraph flow has three nodes: extraction, validation/normalization, and confidence checking. There are no extra agents because the task does not need them.

Whisper is isolated in `app/pipeline/transcription.py`, so replacing local Whisper with the hosted API later would not require changing routes or schema code.

MongoDB access is async through Motor. The client is created in one place and helper functions handle invalid ObjectIds, missing documents, unavailable MongoDB, and driver failures.

## Limitations

The parser depends on transcript quality. If Whisper mishears a number or medical term and the extractor cannot tie a value clearly to the transcript, the field should be empty or the confidence check should fail.

The confidence system is intentionally simple. It is enough to block unreliable extractions and report affected fields, but it is not a calibrated clinical confidence model.

Local Whisper can be slow on long recordings, especially with larger models. The upload path streams to disk, but the transcription library still controls its own processing behavior.
