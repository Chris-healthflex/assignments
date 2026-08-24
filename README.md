# Voice/Note → Structured Clinical Assessment

Turns a clinician–patient WAV recording into a structured `FirstAssessment` JSON that
the production frontend consumes. Pipeline: **WAV → local Whisper → LangGraph
extraction → FirstAssessment (Pydantic v2) → grounding → confidence/flagging → FastAPI
→ MongoDB.**

The design goal is reliability, not extra models: the "advanced" parts are the
**grounding** and **confidence/flagging** stages that make the output trustworthy and
guarantee we *never hallucinate clinical values, scores, or dates*.

---

## Stack

| Concern        | Choice                                            |
|----------------|---------------------------------------------------|
| Transcription  | Local Whisper (`faster-whisper` default / `openai-whisper`) |
| Extraction     | LangGraph agent, one node per clinical section    |
| LLM            | Ollama (local) — deterministic stub fallback      |
| Schema         | Pydantic v2, `extra="forbid"`                      |
| API            | FastAPI (4 endpoints)                              |
| Persistence    | MongoDB (Motor) — in-memory fallback for dev/test  |

---

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-optional.txt        # Whisper + test tooling
cp .env.example .env
```

Put the provided recording at `data/clinical_assessment.wav` (download link is in the
assignment). Then start the local services you want to use:

```bash
# Whisper is local (no service). For the LLM:
ollama pull llama3.1 && ollama serve

# MongoDB (or skip it — see ALLOW_MEMORY_DB below):
docker run -d -p 27017:27017 --name mongo mongo:7
```

### Run the API

```bash
uvicorn app.main:app --reload
# docs at http://localhost:8000/docs
```

### Run the pipeline on the WAV (D5)

```bash
python scripts/run_pipeline.py data/clinical_assessment.wav
# transcript only:
python scripts/transcribe.py data/clinical_assessment.wav
```

No Ollama / Mongo installed? Run everything deterministically:

```bash
USE_STUB_LLM=1 python scripts/run_pipeline.py --transcript "left knee flexion of 124 ..."
```

### Tests

```bash
USE_STUB_LLM=1 pytest        # 35 tests, no external services required
```

---

## Endpoints

| # | Method & path                     | Purpose                                          |
|---|-----------------------------------|--------------------------------------------------|
| 1 | `POST /transcribe-assess`         | Upload a WAV → full pipeline → JSON (`?save=true` to persist) |
| 2 | `POST /assessments`               | Save a parsed result to MongoDB → `{id}`         |
| 3 | `GET  /assessments/{id}`          | Retrieve a saved assessment                      |
| 4 | `GET  /assessments`               | List all; `?start_date=&end_date=` (ISO) filter  |

`GET /health` reports the active DB / LLM / Whisper backends.

---

## Output shape

```
{
  "assessment":   FirstAssessment,        // the 7-section contract
  "transcript":   { text, language, durationSeconds, segments, model, backend },
  "confidence":   { overall, threshold, meetsThreshold, sectionScores, rejectedCount },
  "flaggedFields":[ { path, reason, detail } ],   // reason: not_stated | ungrounded
  "timings":      { <stage>: seconds }
}
```

`data/sample_output.json` is a full example produced by the pipeline.

---

## Design decisions

**One node per section, not one mega-prompt.** The LangGraph agent fans the transcript
out to focused nodes (clinical details, subjective, objective, goals, plan). A weak or
empty section can't corrupt the others, per-section timing is free, and prompts stay
small and testable. If `langgraph` isn't installed the identical nodes run through a
tiny sequential runner — same output, no hard dependency.

**Every leaf is a string defaulting to `""`.** A field we couldn't extract is empty,
never a guessed value. `extra="forbid"` makes any unexpected key from the model a loud
error instead of silent frontend drift. `None`/numbers from the LLM are coerced to
strings so the contract always holds.

**Objective measurements are one row per test, `left`/`right` in that row.** "Left knee
flexion 124° vs 130° on the right" is a single `ObjectiveTest` (`left="124"`,
`right="130"`), not two rows. Counts stay defensible — five stated measurements produce
five rows.

**Grounding is the anti-hallucination gate.** After extraction, every populated value is
checked against the transcript: numbers must appear verbatim, free text must overlap
strongly. Anything unsupported is blanked and recorded as an `ungrounded` flag, so a
fabricated angle or date can't reach the frontend even if the model emits one.

**Confidence + flagging make gaps explicit.** Each section gets a completeness score;
the overall is their mean against a configurable threshold (`meetsThreshold` lets a
caller reject or route for human review). Every empty scalar / empty list is surfaced in
`flaggedFields` as `not_stated`, and ungrounded blanks as `ungrounded` — so a reviewer
sees exactly what was missing vs. what was rejected.

**Runs with zero external services.** `USE_STUB_LLM=1` swaps Ollama for a deterministic,
transcript-grounded rule-based extractor, and `ALLOW_MEMORY_DB=1` swaps Mongo for an
in-process store with the same async surface. This keeps the demo script and the whole
test suite green in CI without a model server or database, while the real Ollama +
MongoDB paths are the defaults in production.

---

## Layout

```
app/
  schemas/assessment.py     FirstAssessment — the contract
  api/                      routes (4 endpoints), request/response schemas, deps
  transcription/            audio validation + Whisper service
  extraction/               LangGraph graph, prompts, llm (ollama+stub),
                            normalizer, grounding, confidence
  db/                       Mongo client (+ memory fallback), repository
  services/                 pipeline orchestrator + assessment service
scripts/                    run_pipeline.py (D5), transcribe.py
tests/                      schema, audio, extraction, grounding, confidence,
                            normalizer, repository, api, pipeline
data/                       clinical_assessment.wav, sample_output.json
```
