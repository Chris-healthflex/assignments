# Voice → Structured Clinical Assessment (FirstAssessment)

Turns a WAV recording of a clinician–patient first-assessment session into the
`FirstAssessment` JSON consumed by the Stance Health clinician frontend, and
persists it to MongoDB.

```
WAV ──► Whisper ──► LangGraph agent ──► FirstAssessment (Pydantic v2, exact schema) ──► MongoDB
                     extract → normalize → audit
```

Stack: Python 3.10+ · FastAPI · openai-whisper · LangGraph · Pydantic v2 · MongoDB (motor)

---

## Setup

```bash
git clone https://github.com/Chris-healthflex/assignments && cd assignments
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env        # then fill in GOOGLE_API_KEY (or OPENAI_/ANTHROPIC_API_KEY)
```

MongoDB — any of:

```bash
docker run -d -p 27017:27017 --name mongo mongo:7        # local
# or set MONGODB_URI to an Atlas connection string in .env
```

Whisper — default is the **local** `base` model (downloads ~140 MB on first run,
no ffmpeg needed: the WAV is decoded with `soundfile` and resampled in-process).
Set `WHISPER_BACKEND=api` to use OpenAI's hosted `whisper-1` instead (needs an
OpenAI key; the local backend needs none, so it is the default here).

## Run

```bash
uvicorn app.main:app --reload        # http://127.0.0.1:8000/docs
```

## Test script (D5)

```bash
python tests/run_pipeline.py clinical_assessment.wav
python tests/run_pipeline.py clinical_assessment.wav --session-date 2026-09-02   # lets the agent resolve "in 6 weeks"
python tests/run_pipeline.py --transcript-only                                    # Whisper only, no LLM
pytest tests/test_schema.py                                                       # offline schema/guardrail tests
```

Prints the transcript (stderr), the FirstAssessment JSON (stdout), writes it to
`tests/output.json`, and lists every flagged field. Exit code `2` mirrors the API's 422.

## Endpoints

| # | Method & path | Purpose |
|---|---|---|
| EP1 | `POST /assessments/parse` | multipart `file` (WAV) → `FirstAssessment` JSON. Optional form fields: `session_date` (ISO), `save=true` |
| EP2 | `POST /assessments` | body `{ "assessment": FirstAssessment, "meta"?: {...} }` → saved doc (201) |
| EP3 | `GET /assessments/{id}` | retrieve by Mongo ObjectId (404 if missing) |
| EP4 | `GET /assessments?from=…&to=…&limit=&skip=` | list, newest first, filterable by `createdAt` |

```bash
curl -F "file=@clinical_assessment.wav" -F "session_date=2026-09-02" http://127.0.0.1:8000/assessments/parse
curl -X POST http://127.0.0.1:8000/assessments -H 'content-type: application/json' -d "{\"assessment\": $(cat tests/output.json)}"
curl "http://127.0.0.1:8000/assessments?from=2026-09-01T00:00:00Z"
```

### Error handling
* `400` – not a WAV / empty upload / bad date range
* `404` – unknown assessment id
* `422` – **extraction confidence below `CONFIDENCE_THRESHOLD`**, with field-level detail:
  ```json
  { "detail": { "message": "...", "overall_confidence": 0.41,
                "fields": [ { "field": "clinicalDetails.duration", "confidence": 0.2, "reason": "Not stated in transcript" } ] } }
  ```
* `500` – transcription / LLM failure (message included)

## Output schema

`app/schemas.py` is the single source of truth. Keys are verbatim from the brief:

```
clinicalDetails        { clinicalHistory, chiefComplaint, duration }
subjectiveAssessments  [ { testName, conclusion } ]
objectiveAssessment    { tests: [ { testName, unitName, value, left, right, comments } ] }
subjectiveGoals        [ { goalDetails, targetDate } ]
objectiveGoals         [ { goalName, goalCategory, unitName, value, targetDate } ]
recommendation         [ { sessionType, sessionFrequency } ]
patientAdvice          { adviceDetails }
```

Enforced by Pydantic: `extra="forbid"` (no extra/renamed keys), `None` → `""`
for every string, lists are always lists (a `null` list fails validation).

---

## Design decisions

**1. Exact-match guarantee, without losing confidence info.**
The brief says the body must be the schema and nothing else, but also that
unconfident fields must be flagged. I resolved this by keeping two models:
the LLM produces `ExtractionDraft = FirstAssessment + flags[] + overall_confidence`;
the API re-validates `assessment` alone into the strict `FirstAssessment` and
returns that as the body. Flags travel in the `X-Extraction-Flags` /
`X-Extraction-Confidence` response headers (and in `meta` when saved), so the
frontend can highlight fields for review without the payload changing shape.

**2. Three-node LangGraph instead of one LLM call.**
`extract` (structured output) → `normalize` (strict re-validation; on failure,
the error is fed back and the LLM retries once) → `audit` (deterministic
guardrails). The audit node is where the "never hallucinate" rule is enforced
mechanically rather than by prompt alone:
* any digit in a `value` / `left` / `right` field that does not literally
  appear in the transcript is flagged at 0.2;
* `targetDate` values are flagged unless stated verbatim or a `session_date`
  was supplied to resolve relative phrases ("in six weeks");
* empty **core** fields (`chiefComplaint`, `clinicalHistory`) force a 422.

**3. Empty string, never a guess.**
Unknown values are `""` + a flag. This is a clinical product; a blank field a
clinician fills in is safer than a plausible number they might not notice.

**4. Confidence threshold → 422.**
`low_confidence = core field below threshold OR overall_confidence below threshold`.
Non-core gaps (e.g. no goals discussed) only produce flags, because a
legitimate session can simply not cover them.

**5. Whisper without ffmpeg, primed with domain vocabulary.**
`whisper.load_audio` shells out to ffmpeg; instead the WAV is read with
`soundfile` and resampled to 16 kHz mono with `scipy.signal.resample_poly`,
so the service runs on a bare Python install. Backend and model size are env-configurable.

Model size matters here. On the provided recording, `base` produced
"evulsion ACL tear", "ankle dose of flexion" and "Butella mobility"; `medium`
recovered "avulsion", "dorsiflexion" and "tibial condyle". Since extraction is
deliberately forbidden from guessing, transcription quality is the ceiling on
output quality � so `WHISPER_MODEL=medium` is the recommended setting, with
`base` available for fast iteration.

`WHISPER_INITIAL_PROMPT` primes the decoder with generic musculoskeletal
vocabulary (not content from any particular recording). This is what recovers
"patellar mobility" from what `medium` alone heard as "tele-mobility", and it
preserves signed ROM values � "negative 5 degrees" of knee extension is
clinically distinct from "5 degrees", and an unprimed decoder drops the sign.

**6. Storage shape.**
Each document stores `assessment` exactly as returned plus an optional `meta`
audit trail (source filename, transcript, flags, confidence) and `createdAt`
(indexed) for the date filter. `id` and `createdAt` are returned *beside*
`assessment`, never inside it.

**7. Provider-agnostic LLM.** `LLM_PROVIDER=openai|anthropic|google`, temperature 0,
structured output via `with_structured_output` for schema-constrained generation.
The extraction graph never sees provider-specific code, so swapping models is a
one-line `.env` change. Default here is `google` / `gemini-3.5-flash`, chosen over
`gemini-3.6-flash` because the latter is a reasoning model: ~43 s versus ~1.7 s on a
trivial prompt, with no observable benefit on an extraction task this constrained.

## Project layout

```
app/
  main.py           FastAPI app, 4 endpoints, error handling
  schemas.py        FirstAssessment (exact) + internal extraction models
  transcription.py  Whisper (local / API)
  agent.py          LangGraph extract → normalize → audit
  db.py             MongoDB connection + save / get / list
  config.py         env settings
tests/
  run_pipeline.py   D5 end-to-end script
  test_schema.py    offline unit tests
requirements.txt · .env.example
```

## Known limitations / next steps
* Single-speaker transcript (no diarisation). Whisper `small`/`medium` improves
  accuracy on clinical vocabulary at the cost of speed; `base` is the default for CI-friendliness.
* Transcription is the accuracy bottleneck, not extraction. Residual ASR errors
  propagate verbatim into the output by design � the agent copies the transcript
  rather than "correcting" it toward a plausible clinical term, because silently
  rewriting a misheard value is the failure mode that matters in a medical product.
* Measured on the provided recording (all with vocabulary priming):

  | term            | `base`        | `medium`     | `large-v3`  |
  |-----------------|---------------|--------------|-------------|
  | avulsion        | "evulsion"    | correct      | correct     |
  | dorsiflexion    | "dose of flexion" | correct  | correct     |
  | tibial condyle  | "condal"      | correct      | correct     |
  | patellar        | "Butella"     | "Patella"    | "Patellar"  |
  | "negative 5"    | "negic 5"     | "knee gig 5" | "negig 5"   |

  `medium` (~5 min) is the recommended default; `large-v3` (~15 min, 2.9 GB) buys
  only "Patellar" over "Patella". **No model recovers "negative 5 degrees"** for
  right knee extension, so that value is reported as `5` rather than `-5`. This is
  left uncorrected deliberately: the pipeline reports what was said, and inventing
  the sign would be exactly the hallucination the guardrails prohibit. A clinician
  reviewing the form is the correct place to catch it.
* Numeric anti-hallucination check is string-based; spoken numbers ("one hundred and ten")
  are not matched, so they surface as flags rather than silently passing.
* No auth — assumed to sit behind the existing API gateway.
