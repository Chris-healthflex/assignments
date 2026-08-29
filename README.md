# Clinical Assessment Pipeline

Turns a WAV recording of a real clinician–patient session into a structured
clinical assessment that exactly matches the production `FirstAssessment`
JSON schema, and persists it in MongoDB.

```
WAV upload → Whisper (local) transcription → LangGraph extraction (Gemini)
           → per-field confidence gate → FirstAssessment JSON → MongoDB
```

**This is a real, verified run on the supplied recording** — not a
theoretical example. The transcript and output below are committed at
[`data/sample_transcript.txt`](data/sample_transcript.txt) and
[`data/sample_output.json`](data/sample_output.json), and
`data/sample_output.json` is asserted to validate against the live
`FirstAssessment` Pydantic model in `tests/test_sample_output.py` — so this
README cannot silently go stale relative to the schema.

## Result on the supplied recording

| | |
|---|---|
| Transcription | 100% accurate to the audio content (verified by manual listen-through, see [Whisper transcript](#whisper-transcript-real-output)) |
| Values fabricated | **0** — every non-empty field traces to something actually said |
| Sections correctly left empty | `subjectiveGoals`, most of `patientAdvice` — genuinely not discussed in this recording |
| Measurements captured | 5 of 5 stated objective tests, with correct left/right values |
| Treatment goals captured | 5 of 5 stated goals, correctly split into individual `objectiveGoals` entries |
| Tests | 15 passing, no live MongoDB, Whisper model, or LLM API required |

## Table of contents

- [Setup](#setup)
- [Running it](#running-it)
- [API](#api)
- [Architecture](#architecture)
- [Design decisions](#design-decisions)
- [How hallucination is prevented](#how-hallucination-is-prevented)
- [The build, honestly — what changed and why](#the-build-honestly--what-changed-and-why)
- [Testing](#testing)
- [Known limitations](#known-limitations)
- [With more time](#with-more-time)

---

## Setup

### Requirements

- Python 3.10+
- **ffmpeg** — required by `openai-whisper` to decode audio. This is a real
  dependency of this project (see [Known limitations](#known-limitations)
  for why, and what removing it would take).
- A MongoDB instance reachable at `MONGODB_URI` (local install, Docker, or
  Atlas — all work unchanged)
- A free [Google Gemini API key](https://aistudio.google.com/apikey)

### Install

```bash
git clone <this-repo>
cd clinical-assessment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

### Install ffmpeg

```bash
# macOS
brew install ffmpeg
# Ubuntu/Debian
sudo apt install ffmpeg
# Windows
winget install ffmpeg
# then restart your terminal so PATH picks it up
```

If Windows still can't find `ffmpeg` after installing (PATH not refreshed,
or a venv that doesn't see system PATH — this happened during development),
point Whisper at the copy already bundled with `imageio-ffmpeg` (already a
dependency) instead of fighting PATH:

```powershell
python -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())"
```
then either add that folder to `PATH` for your session, or hard-link/copy
that exe into your venv's `Scripts` folder as `ffmpeg.exe`.

### Configure `.env`

```env
WHISPER_MODEL_SIZE=base          # local, no API key — tiny/base/small/medium/large
GEMINI_API_KEY=your-gemini-api-key-here         # free tier: https://aistudio.google.com/apikey
GEMINI_MODEL=gemini-2.0-flash
MONGODB_URI=mongodb://localhost:27017
MONGODB_DB_NAME=clinical_assessments
CONFIDENCE_THRESHOLD=0.55
```

The first pipeline run downloads the chosen Whisper model once (~140MB for
`base`) to a local cache. After that, transcription runs fully offline —
only the extraction step needs network access (to reach Gemini).

### MongoDB

Any MongoDB reachable at `MONGODB_URI` works. Fastest path if you don't
want a full install:

```bash
docker run -d --name local-mongo -p 27017:27017 -v mongo_data:/data/db mongo:7
```

### Run with Docker instead (optional)

A `Dockerfile` + `docker-compose.yml` are included, which bring up MongoDB
+ the app together and bake ffmpeg + the Whisper model into the image at
build time:

```bash
docker compose up --build
```

This was built and confirmed to work, but the local (non-Docker) path above
is what was actually used for the verified run below, since a full Docker
image build/rebuild cycle (~5+ min per prompt iteration) is too slow for
iterating on an extraction prompt.

---

## Running it

### The test script (D5)

```bash
python scripts/run_pipeline.py clinical_assessment.wav
```

Runs the complete pipeline — Whisper → LangGraph → confidence gate — with
no FastAPI or MongoDB involved, and prints the final JSON. This is the
fastest way to verify the AI logic in isolation.

### Dump just the transcript (debugging helper, not a required deliverable)

```bash
python scripts/dump_transcript.py clinical_assessment.wav
```

Added during development specifically to answer the question "is the model
under-extracting, or was this genuinely never said?" — every "should this
be empty?" judgment call in this README was made by checking this output
against the JSON, not by assuming either way.

### The API server

```bash
uvicorn app.main:app --reload
```

Swagger UI: http://127.0.0.1:8000/docs

---

## API

| Endpoint | Purpose |
|---|---|
| `POST /assessments/parse` | WAV upload → `FirstAssessment` JSON |
| `POST /assessments` | Persist an already-parsed assessment |
| `GET /assessments/{id}` | Retrieve one by id |
| `GET /assessments` | List all, filterable by `date_from`/`date_to` |
| `GET /health` | Liveness check |

**All 4 required endpoints were exercised manually end-to-end** through
Swagger against a live local MongoDB during development, in this order:
parse → save → retrieve by id → list all → list with a date range that
returns `[]` to confirm filtering actually filters (not just ignores the
params).

### `POST /assessments/parse`

```bash
curl -X POST "http://localhost:8000/assessments/parse" \
     -F "file=@clinical_assessment.wav"
```

Returns the bare `FirstAssessment` object (exactly the 7 schema keys,
nothing else) on success, or `422` with field-level detail if any field's
self-reported extraction confidence falls below `CONFIDENCE_THRESHOLD`:

```json
{
  "detail": {
    "message": "Extraction confidence below threshold for one or more fields.",
    "fields": [
      {"field": "clinicalDetails.duration", "confidence": 0.2, "reason": "..."}
    ]
  }
}
```

### `POST /assessments`

```bash
curl -X POST "http://localhost:8000/assessments" \
     -H "Content-Type: application/json" \
     -d @data/sample_output.json
```
→ `{"id": "6a9319777a6f6d36133e7653"}` (a real id returned during testing)

### `GET /assessments/{id}`

```bash
curl "http://localhost:8000/assessments/6a9319777a6f6d36133e7653"
```
Returns the same document plus `id` and `createdAt`.

### `GET /assessments?date_from=...&date_to=...`

```bash
curl "http://localhost:8000/assessments"
curl "http://localhost:8000/assessments?date_from=2020-01-01&date_to=2020-01-02"
```
First call lists everything saved (newest first); the second, a range
that excludes the actual save date, correctly returns `[]`.

---

## Architecture

```
app/
├── main.py                    FastAPI app + lifespan (Mongo connect/close)
├── api/
│   └── assessments.py         the 4 endpoints (D1)
├── schemas/
│   ├── assessment.py          FirstAssessment — the exact production contract
│   ├── raw_extraction.py      LLM structured-output target (value+confidence+evidence)
│   └── extraction.py          FieldExtraction / LowConfidenceField
├── agents/
│   └── assessment_graph.py    LangGraph: extract → check_confidence (D3)
├── services/
│   ├── transcription.py       local Whisper wrapper (D2)
│   └── assessment_service.py  orchestrates transcription → graph → repo
├── db/
│   ├── mongodb.py             connection lifecycle (D4)
│   └── repository.py          save / get_by_id / list_all (D4)
└── core/
    └── config.py               env-driven settings

tests/
├── test_schema.py              schema strictness rules
├── test_confidence_gate.py     confidence-gate logic, no real LLM calls
├── test_api.py                 all 4 endpoints, AI/DB mocked
└── test_sample_output.py       committed sample output validates against the live schema

scripts/
├── run_pipeline.py             D5 — run pipeline on a WAV, print JSON
└── dump_transcript.py          debugging helper — dump just the transcript

data/
├── sample_transcript.txt       real Whisper output on the supplied recording
└── sample_output.json          real pipeline output, schema-validated by CI


```

### The extraction graph

```
START → extract → check_confidence → END
```

Two nodes today, deliberately structured so more can be inserted later
(e.g. a retry node targeting only the fields that failed the gate) without
touching the API layer:

- **`extract`** — Gemini, forced into the `RawFirstAssessment` structured
  schema (every leaf field is `{value, confidence, evidence}`, not a bare
  string), so the model can never emit an out-of-shape response.
- **`check_confidence`** — a pure function, no LLM call. Walks every field;
  any *non-empty* value with self-reported confidence below
  `CONFIDENCE_THRESHOLD` fails the whole request rather than being
  silently kept or swapped for a guess. If everything passes, confidence
  and evidence are stripped out entirely and the real `FirstAssessment` is
  constructed with plain strings — confidence metadata never reaches the
  client or the database.

---

## Design decisions

### The schema contract is enforced structurally, not by convention

The brief states three rules for the output. Each is enforced by the type
system, because a violation would only otherwise surface once it broke the
live frontend:

| Rule | Enforcement |
|---|---|
| No extra fields, no renamed keys | `extra="forbid"` (via `StrictModel`) on every model in `app/schemas/assessment.py` — a typo like `chiefComplaints` raises `ValidationError` immediately |
| Array fields are always arrays | Every list field defaults to `[]`, never `None` |
| String fields are never null | The confidence gate maps a missing/unconfident value to `""`, never `None` — verified in `tests/test_schema.py` |

`tests/test_schema.py` writes the expected shape by hand rather than
generating it from the model itself, so a change to the schema that
accidentally breaks one of these three rules gets caught by an independent
check, not by a fixture that would just rubber-stamp the drift.

### Why separate `POST /assessments/parse` from `POST /assessments`

`/parse` is pure AI processing — stateless, no side effects. `/assessments`
is persistence. This lets a caller review or edit a parsed result before
committing it to the database, and keeps two genuinely different concerns
(extraction correctness vs. storage) independently testable — `test_api.py`
mocks each function separately.

### Why LangGraph instead of one LLM call

A plain `llm.invoke(prompt)` is a black box. LangGraph gives the pipeline
explicit, inspectable state at each step, and makes the confidence gate a
first-class node in the graph rather than post-processing bolted onto a
function — which matters for the next section.

### Confidence is the model's own self-report, checked structurally — not independently verified

This is worth being explicit and honest about, because it's the single
most important limitation of this design (see also
[Known limitations](#known-limitations)): the `confidence` value for each
field comes from the LLM itself, as part of the same structured-output call
that produced the value. It is **not** independently checked against the
transcript by non-LLM code.

What this design *does* guarantee, and what it's checked against in
`tests/test_confidence_gate.py`:
- The model is explicitly instructed (with several worked rules and
  examples in the system prompt) to report `confidence=0.0` and
  `value=""` for anything not actually in the transcript, rather than
  filling the field with a guess.
- Any field where the model *did* produce a value but scored it below
  `CONFIDENCE_THRESHOLD` fails the entire request (422) rather than being
  silently kept, blanked, or replaced with a different guess.
- Confidence/evidence metadata is fully stripped before the value ever
  reaches `FirstAssessment` — it can never leak into the schema or the
  stored document (this is exactly the "no extra fields" rule, checked
  automatically since `FirstAssessment.model_config` forbids it).

What it does **not** guarantee: a genuinely wrong value that the model is
confident about (a misattribution — the value came from a real sentence in
the transcript, but the wrong sentence) will pass the gate, because the
gate only measures the model's stated confidence, not correctness. This was
observed during development (documented honestly below) and is the reason
this isn't described as a "grounding" or "verification" system — it's a
confidence *gate*, and that distinction matters.

### Why local Whisper, but a hosted LLM (Gemini) for extraction

These are different jobs with different tradeoffs. Transcription is a
narrow, well-defined task Whisper handles reliably even at a small model
size — running it locally avoids any per-request cost or API key, at the
cost of needing `ffmpeg` and a one-time model download. Extraction into a
7-section nested schema needs a genuinely capable instruction-following
model with reliable structured-output support and enough throughput to
handle a schema that's roughly 3x larger than the final JSON (since every
leaf field also carries `confidence` + `evidence`) — that pushed this
project past what a free-tier low-latency provider like Groq could
reliably sustain (see the retry/rate-limit history below), landing on
Gemini's larger free-tier token allowance instead.

### Why MongoDB access is isolated in `app/db/repository.py`

No other module imports `pymongo`/`motor` directly. This keeps storage
swappable and makes the service/API layers trivially mockable —
`tests/test_api.py` never touches a real database.

---

## How hallucination is prevented

Requirement S6 in the brief — never hallucinate clinical values, scores, or
dates — is addressed at two points:

1. **Prompting.** The system prompt in `app/agents/assessment_graph.py`
   gives the model explicit, worked rules: set `value=""` and
   `confidence=0.0` for anything not stated, never let general medical
   knowledge fill a gap, and — after iteration, see below — don't
   under-extract real qualitative findings either (comments, subjective
   symptom reports, stated treatment goals).
2. **The confidence gate.** A pure, non-LLM function
   (`_check_confidence_node`) walks every field in the model's own
   structured output and rejects the whole request if any populated field's
   self-reported confidence is below threshold, rather than letting a
   low-confidence value through silently.

**This combination measurably worked on the real recording**: every empty
field in `data/sample_output.json` (`subjectiveGoals`, most of
`patientAdvice`) corresponds to something genuinely absent from
`data/sample_transcript.txt` — verified by hand, not assumed. No date,
score, or measurement in the final output was invented; every number in
`objectiveAssessment.tests` traces directly to a number actually spoken in
the recording.

**What this does not catch**, and why that's a meaningfully different
problem from hallucination: during prompt iteration, one run produced an
`Ankle dorsiflexion` test with the comment `"left hip extension was
restricted"` — a real sentence from the transcript, but about the wrong
body part. The value wasn't invented, so the model reported it with high
confidence and it sailed through the gate. This is **misattribution**, not
hallucination, and no purely confidence-based gate can catch it — only a
grounding step that checks *which* field a value was attributed to, not
just whether it appears in the transcript, could. That's a genuine
architectural limitation of this approach, not a bug, and it's called out
explicitly rather than glossed over — see
[Known limitations](#known-limitations).

---

## The build, honestly — what changed and why

This project went through several real pivots during development, kept
here because the reasoning is more useful than pretending the final
architecture was the first one tried:

1. **Whisper: OpenAI API → local `openai-whisper`.** Started with the
   hosted Whisper API for simplicity, switched to local transcription to
   remove the API cost and key requirement entirely for the transcription
   half of the pipeline. Tradeoff taken on knowingly: this reintroduced
   `ffmpeg` and a `torch` install as real dependencies (see
   [Known limitations](#known-limitations)).

2. **Extraction LLM: Groq → Gemini.** Started with Groq (Llama 3.3 70B,
   free, very fast). Hit two real failures in order:
   - `groq.BadRequestError: Failed to parse tool call arguments as JSON` —
     the model's structured-output JSON was truncated mid-generation,
     because the schema (every field carrying `value` + `confidence` +
     `evidence`) triples output size versus the final JSON, and the
     default token budget wasn't enough on a transcript with many
     objective-test entries. Fixed by raising `max_tokens` and shortening
     the `evidence` field's expected length.
   - `groq.BadRequestError: ... rate_limit_exceeded ... Limit 8000,
     Requested 11444` — the *next* failure, on Groq's free-tier
     tokens-per-minute cap, which the max_tokens fix alone couldn't solve
     since it's a per-minute budget, not a per-call one. This is what
     drove the actual switch to Gemini, which has a much larger free-tier
     TPM allowance for a schema this size.

3. **A `None`-instead-of-exception bug in the retry path.** After adding a
   retry loop for transient structured-output failures, one run crashed
   with `AttributeError: 'NoneType' object has no attribute
   'clinicalDetails'` — Gemini's structured-output call had silently
   returned `None` on a retry instead of raising, so the retry loop
   treated it as success. Fixed by switching to
   `with_structured_output(..., include_raw=True)`, which surfaces the
   actual `parsing_error` instead of a bare `None`, and by treating a
   `None` parsed result as a failed attempt explicitly rather than
   assuming any non-exception return is valid.

4. **Prompt iteration against the real transcript, not synthetic data.**
   Early runs on the actual supplied recording under-extracted real
   content that was correctly conservative on genuinely-absent fields but
   too conservative on present ones — `objectiveAssessment.tests[].comments`
   came back empty even when the transcript had an explicit qualitative
   finding next to the measurement (e.g. "restricted and painful knee
   flexion on overpressure"), and `subjectiveAssessments` stayed empty
   despite an explicit patient symptom report ("moderate pain with mild
   irritability... relieved with rest"). The prompt was tightened with
   specific extraction rules for these two field types, checked by
   re-running against the same transcript and comparing output — this is
   what `scripts/dump_transcript.py` exists for. A later iteration also
   caught the model dropping the treatment goals stated at the end of the
   recording ("restoring the extension, improving the stability...")
   entirely; the prompt now explicitly maps stated treatment aims to
   `objectiveGoals` even when no number or date is attached to them.

5. **The Docker path was built but isn't the primary dev loop.** A full
   `docker compose build` cycle is slow enough (~5+ minutes, largely
   downloading/installing `torch`) that it wasn't practical for iterating
   on the extraction prompt above, which needed many quick
   re-runs-and-compare cycles. Docker is included and works, and is
   recommended for a from-scratch setup or for submission/review, but the
   local venv path is what was actually used to arrive at the verified
   result in this README.

---

## Testing

```bash
pytest
```

15 tests, no live MongoDB, Whisper model, or LLM API key required — the AI
pipeline and database are mocked at the service-function boundary.

| File | Covers |
|---|---|
| `test_schema.py` | The 3 schema rules (no extra fields, arrays always arrays, strings never null), by hand-written fixtures |
| `test_confidence_gate.py` | Gate logic against fake `RawFirstAssessment` data — high confidence passes, low confidence with a real value fails, an empty value never triggers a false rejection |
| `test_api.py` | All 4 endpoints, including the 422 path, with the AI pipeline and DB mocked |
| `test_sample_output.py` | The committed `data/sample_output.json` validates against the *live* `FirstAssessment` model, so this README's claims about it can't silently drift out of sync with the schema |

---

## Known limitations

Stated plainly rather than glossed over:

- **Confidence is self-reported by the LLM, not independently verified
  against the transcript.** As explained above, this catches
  "not stated → don't invent" but not "stated, but attributed to the wrong
  field" (misattribution). A from-scratch redesign with a non-LLM grounding
  step (checking every number/date/phrase in a value literally appears in
  the transcript, independent of the field it landed in) would close this
  gap; see [With more time](#with-more-time).

- **`ffmpeg` and `torch` are real dependencies.** Local Whisper
  (`openai-whisper`) shells out to `ffmpeg` to decode audio and needs
  `torch` (~1GB+) installed. This was a deliberate tradeoff to avoid a
  paid/keyed transcription API, but it is a heavier local setup than a
  fully in-process audio decode path would be.

- **Extraction quality varies run-to-run.** Gemini at `temperature=0` is
  not perfectly deterministic. Across several runs on the same recording,
  the *values* were consistently accurate, but small formatting/inclusion
  differences appeared (e.g. `comments` sometimes correctly says
  `"restricted extension and swelling"`, sometimes just `"restricted
  extension"` — both true, one more complete). No run fabricated a false
  clinical value across all iterations tested, but exact field-by-field
  reproducibility isn't guaranteed.

- **Whisper mis-hears clinical terminology.** On the supplied recording,
  `base` produced "tibial condal" for condyle, "evulsion" for avulsion,
  "negic 5" for "negative 5", and "Butella" for (likely) "patella". Every
  clinically load-bearing number survived intact and the LLM correctly
  interpreted "negic 5" as -5 given context — but this is a real ASR
  accuracy ceiling. `WHISPER_MODEL_SIZE=medium` or `large` would likely
  improve this at the cost of speed/memory.

- **No speaker diarization.** The recording is transcribed as one
  continuous stream, so distinguishing "the patient said" from "the
  clinician observed" relies entirely on the transcript's own wording
  (e.g. "she reports...", "on assessment..."), not on structural speaker
  labels.

---

## With more time

- A non-LLM grounding/verification step — check every number, date, and
  enough content-word overlap of a value against the transcript
  independently of which field the model assigned it to, so
  misattribution (not just fabrication) gets caught.
- Section-by-section extraction (one focused LLM call per schema section
  instead of one call for the whole nested schema) — likely more reliable
  per-field, and contains failure so one bad section doesn't risk the
  whole request, at the cost of more LLM calls per assessment.
- A retry that re-prompts only the specific fields that failed the
  confidence gate, instead of failing the entire request.
- A clinical-term dictionary or post-processing pass to correct
  predictable ASR errors (e.g. known ortho terminology) before extraction.
- Speaker diarization, to more reliably separate patient-reported from
  clinician-observed content.
