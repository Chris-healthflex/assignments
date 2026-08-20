# V2N — Voice to Note (First Assessment)

Turns a recorded physiotherapy first-assessment consultation into a validated,
structured `FirstAssessment` document stored in MongoDB.

```
audio ──▶ Whisper ──▶ transcript ──▶ LangGraph agent ──▶ FirstAssessment ──▶ MongoDB
                                     (extract → validate → repair)
```

## Layout

| Path | Responsibility |
|---|---|
| [app/main.py](app/main.py) | FastAPI app: `/transcribe`, `/extract`, `/assessments` |
| [app/schemas.py](app/schemas.py) | Pydantic `FirstAssessment` + extraction payload models |
| [app/transcription.py](app/transcription.py) | Whisper (faster-whisper, local by default) |
| [app/extraction.py](app/extraction.py) | LangGraph agent with a bounded repair loop |
| [app/db.py](app/db.py) | Motor client, save/retrieve, indexes |
| [app/config.py](app/config.py) | Settings from env / `.env` |
| [tests/test_schema.py](tests/test_schema.py) | Schema contract tests (no services needed) |
| [tests/run_pipeline.py](tests/run_pipeline.py) | D5 end-to-end script |

## Design notes

- **Every clinical field is optional.** A consultation that never mentions
  medication must not produce a validation error, and the agent must not invent
  one. Fields the transcript does not support are listed in
  `meta.unresolved_fields` instead of being guessed.
- **`extra="forbid"` on every model.** Schema drift fails loudly at the boundary
  rather than silently writing junk into Mongo.
- **The LLM never owns identity or provenance.** It produces an
  `ExtractionPayload`; ids, timestamps and `meta` are set by our code.
- **Repair, not retry.** When validation fails, the specific errors are fed back
  to the model (bounded by `EXTRACTION_MAX_RETRIES`) rather than re-rolling the
  same prompt.
- **LangGraph orchestrates; the official Anthropic SDK calls the model.** Keeps
  the model call surface first-party and the graph logic explicit.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                 # then set ANTHROPIC_API_KEY

docker compose up -d mongo
uvicorn app.main:app --reload
```

Then:

```bash
curl -F "file=@sample.wav" http://localhost:8000/transcribe
curl -X POST http://localhost:8000/extract \
     -H 'content-type: application/json' \
     -d '{"transcript": "..."}'
```

## Tests

```bash
pytest                                        # schema contract tests
python -m tests.run_pipeline sample.wav       # D5 end-to-end
python -m tests.run_pipeline --transcript t.txt   # skip Whisper
```

## Status

Phase 0 scaffold. Module boundaries, schema shape and the agent graph are in
place; `FirstAssessment` field names still need to be reconciled with the
canonical assignment spec.
