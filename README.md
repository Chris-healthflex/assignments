# First Assessment

Turns a clinician–patient consultation recording into a structured
`FirstAssessment` document, stores it in MongoDB, and serves it over a REST API.

Transcription runs locally with Whisper. Extraction runs through a LangGraph
agent that must quote the transcript for every value it produces, and every
quote is verified in our own code before the value is returned.

---

## What this achieves

**All six deliverables, working end to end against the real recording and a live
MongoDB Atlas cluster.**

| # | Deliverable | Where |
|---|---|---|
| D1 | FastAPI service | [`app/main.py`](app/main.py) — 4 endpoints + `/health` |
| D2 | Whisper transcription module | [`app/transcription.py`](app/transcription.py) — local, word-level, disk-cached |
| D3 | LangGraph agent with Pydantic output | [`app/extraction.py`](app/extraction.py) — 3-node fan-out, grounding, repair loop |
| D4 | MongoDB models, connection, save/retrieve | [`app/db.py`](app/db.py), [`app/schemas.py`](app/schemas.py) |
| D5 | Test script printing JSON | [`tests/run_pipeline.py`](tests/run_pipeline.py) |
| D6 | README with setup and design decisions | this file |

### A full run, through the live API

```
POST /assessments/parse   422    46 fields extracted, 0 unsourced, overall 89%
POST /assessments         201    id 6a875278f35f46853c5b0691
GET  /assessments/{id}    200    contract identical: True
GET  /assessments?date=   200    2026-08-20 → 1,  2020-01-01 → 0

bad id → 404   unknown id → 404   bad date → 422   pdf upload → 415
```

Measurements extracted from the recording, every one traced to a quote:

```
knee flexion           L 124   R 130   degrees
knee extension         L 20    R 5     degrees   ← flagged at 0.05, see Problems
hip internal rotation  L 45    R 45    degrees
hip external rotation  L 60    R 60    degrees
ankle dorsiflexion     L 4.5   R 12    degrees
```

### What is verifiably true about the output

- **The contract is exact.** Seven sections, in order, no extra keys, no renamed
  keys, arrays that stay arrays, strings that are never null — checked against
  the serialised JSON on the wire, not against the model.
- **Nothing is invented.** Every `targetDate` is `""`, because the recording
  states no dates. `objectiveGoals` and `patientAdvice` are empty, because the
  clinician set no numeric targets and gave no advice. Empty is a correct answer
  here, and the system prefers it to a plausible guess.
- **Low-confidence values are held back**, per field, with the reason and the
  evidence behind them.
- **195 automated checks pass** — 137 Python tests (2 skip without a cached
  transcript) and 58 frontend checks.
- **WAV, MP3 and M4A** were each transcoded from the same source and produced
  identical transcripts.

---

## Quick start

Requires **Python 3.14** (built and tested on 3.14.3), a **Google AI Studio API
key** (free), and a **MongoDB** instance (Atlas free tier is fine).

```bash
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env            # then fill in GOOGLE_API_KEY and MONGODB_URI
uvicorn app.main:app --reload
```

Open **http://localhost:8000/ui/** for the review interface, or **/docs** for the
OpenAPI explorer.

Place the consultation recording at `./clinical_assessment.wav`, then:

```bash
python -m tests.run_pipeline clinical_assessment.wav          # in-process
python -m tests.run_pipeline clinical_assessment.wav --http   # through the API
python -m tests.run_pipeline clinical_assessment.wav > out.json
```

**Three things to expect on a first run:**

1. **Whisper downloads itself** — ~1.5 GB of model weights into
   `~/.cache/huggingface/hub`, once. Nothing is committed to this repo.
2. **The first transcription takes minutes**, not seconds — roughly 3 minutes of
   CPU for the 105-second sample. The result is cached by content hash, so every
   run after that is instant.
3. **Two tests skip** until you have transcribed something. They exercise the
   cache round-trip and print the command that satisfies them. Expected on a
   fresh clone, not a failure.

The audio file is **not** in this repository and `*.wav` is **not** in
`.gitignore` — check `git status` before committing if you drop a recording in.

---

## How it works

```
  WAV / MP3 / M4A
        │
        ▼
  ┌───────────────┐   faster-whisper, run locally, word_timestamps=True
  │ Transcription │   → text plus a probability for every single word
  └───────┬───────┘   → cached on disk by a hash of the audio bytes
          │
          ▼   ┌─> extract_subjective ─┐
        START ┼─> extract_objective ──┼─> assemble ─> ground ─┐
              └─> extract_plan ───────┘        ▲              │
                                               │              ▼
                                            repair <──── still ungrounded?
                                                              │ no
          ┌───────────────────────────────────────────────────┘
          ▼
  assessment + per-field evidence
          │
          ├─> any field below the threshold, or a section missing?  ─> 422
          │
          └─> POST /assessments  ─>  MongoDB
```

**Three concurrent calls, not one.** The seven sections are grouped into
`subjective` (clinical details, subjective assessments), `objective`
(measurements) and `plan` (goals, recommendation, advice). Each is a separate
structured-output call, run in parallel.

**Every value must cite its source.** The model returns each value together with
a verbatim quote from the transcript. `ground()` — plain Python, no model
involved — then checks that the quote actually appears in the transcript and
looks up how confidently Whisper heard those exact words. **The model never gets
a vote on whether its own output is trustworthy.**

**A value that cannot be traced is repaired, then flagged.** Ungrounded fields
send the graph back through `repair`, which re-asks only the offending group and
names the bad quote. After two attempts, whatever is still unverified is reported
rather than quietly kept.

### The schema contract

The brief's three rules are enforced as code, not convention:

| Rule | How it is enforced |
|---|---|
| No extra fields, no renamed keys | `extra="forbid"` on every model — a renamed key arrives as an unknown key and raises |
| Arrays stay arrays, even with one item | Every list defaults to `[]`; a `None` is normalised to `[]`, never dropped |
| String fields are strings, never null | Every leaf is a `CleanStr`, which maps `None` → `""` at the boundary |

Two details worth naming:

- **Field names are declared in camelCase literally**, not generated by
  `alias_generator=to_camel`. With aliases, a plain `model_dump()` silently emits
  snake_case and only `model_dump(by_alias=True)` is correct — one forgotten flag
  anywhere writes a wrong-shaped document. Declaring camelCase makes the correct
  output the *only* possible output.
- **`recommendation` is singular but holds an array.** That is the frontend's
  spelling. "Fixing" it would be a renamed key, which the brief forbids.

### The four endpoints

```
POST /assessments/parse      multipart audio  ->  draft assessment, or 422
POST /assessments            draft            ->  saved, with an id
GET  /assessments/{id}       id               ->  one assessment
GET  /assessments?date=      YYYY-MM-DD       ->  that day's assessments
```

All four speak one envelope: the untouched `FirstAssessment` under `assessment`,
with transcript, confidence and identifiers wrapped around it. A wrapper is
unavoidable — the brief forbids extra fields *inside* the contract and requires
confidence to be returned *with* the result, so the confidence has to live
outside it. Given that, one envelope used identically everywhere beats four
different response shapes.

---

## What changed along the way, and why

Each of these replaced something that was already working. They are recorded
because the reasoning matters more than the result.

### Whisper `small` → `medium`

`small` transcribed the sample recording as **"ankle dosa flexion"** and
**"left-to-be-all-condylo fracture"**. `medium` gives **"ankle dorsiflexion"** and
**"tibial condyle fracture"**.

The cost is real — 184 seconds versus 24 on CPU — but it is paid **once per
recording**, because transcripts are cached by a hash of the audio bytes plus the
decode settings. A mangled clinical term, by contrast, is paid on every read of
that assessment forever.

### Confidence per conversation → confidence per field

The first design scored the extraction as a whole. That hides exactly the case
that matters: one destroyed measurement inside an otherwise clean assessment
disappears into an average. Scoring every field independently is what makes the
`422` specific enough to act on, and it reshaped the schema — evidence records,
not a single number.

### Seven extraction nodes → three

The first version had one node per section. That fired seven concurrent calls and
exhausted a free-tier daily quota in three runs. Grouping into subjective /
objective / plan cut it to three calls per extraction with no loss of focus — the
groups match what a clinician reasons about together.

### `gemini-2.5-flash` → `gemini-3.1-flash-lite`

The 2.5 and 3.7 flash tiers allow 5 requests/minute and a few dozen per day on
the free tier, which two full runs exhaust. `flash-lite` has real headroom
(15/min) and handled the citation discipline without difficulty. Client-side
pacing was added alongside it, held one under the limit rather than at it — the
limiter can pace what we send, but not the retries the SDK fires on its own.

### Scoring the quote → scoring the quote *and its neighbourhood*

A live run exposed something worth recording. Told to "quote the shortest span
that establishes the value", the model quoted `"5 degrees on the right"` — every
word of which Whisper heard at 93% or better — stepping neatly over the 5% word
beside it. **A perfectly well-behaved model had steered around the safety check.**

The fix was a ±3-word window around every quote. That initially produced false
alarms on misheard function words — a mangled "and" between two goals casts doubt
on neither — so `CONTEXT_IGNORED` excludes them. A misheard "negative" before a
measurement still counts.

### Motor → PyMongo's `AsyncMongoClient`

Motor reached end-of-life in May 2026 once PyMongo 4.13 absorbed the async API.
Starting a new project on a driver with no upstream is not defensible. The API is
near-identical; the only visible difference is the import.

### A silent partial failure → `422`, and a total one → `502`

The extraction makes three concurrent calls. When one failed, its sections came
back empty — and an empty section normally means the clinician did not mention
it, which is a *correct* answer. The two were indistinguishable in the finished
document.

Worse, confidence *rose* when a section was lost, because it averages only the
fields that came back. One run returned 90% overall with a third of the
assessment silently missing.

Three changes fixed it:

1. **A group that returned nothing is now retried.** The repair loop originally
   looked only at fields whose quotes did not check out — a failed call produced
   no fields, so it was invisible to that loop and never asked again. A bad quote
   got two more chances; a dropped connection got none.
2. **Surviving failures name the sections they cost**, in `flags.failedSections`,
   by contract section rather than internal group name.
3. **The status code stopped lying.** A partial failure is a `422` carrying a
   `section_unavailable` entry per lost section — *even when every field that did
   return scored perfectly*. Total failure is a `502`: nothing was wrong with the
   request, and "try again" is the useful advice, not "fix your input".

### Other decisions worth stating

- **Whisper runs locally, and the hosted backend deliberately raises
  `NotImplementedError`.** This is patient audio; shipping PHI to a third party
  is a decision for a compliance review, not a default in a config file.
- **Temperature is 0.0**, not as a style preference: extraction must be
  reproducible, and sampling is a source of invented detail.
- **Confidence is recomputed on read, never stored.** Only the raw signals go to
  MongoDB. A derived value in a database goes stale the moment the scoring rule
  changes; the inputs do not.
- **Saving does not re-apply the confidence gate.** The gate belongs where a
  *machine* produces values. `POST /assessments` receives what a human has
  already reviewed — gating it would mean a clinician who corrected a misheard
  measurement could not save the correction.
- **`?date=` filters the envelope's `createdAt`, in UTC.** `FirstAssessment` has
  no date of its own; its only dates are goal `targetDate` values, which are
  intentions rather than a record of when anything happened. The range is
  half-open `[midnight, next midnight)` — BSON keeps milliseconds, so an
  inclusive upper bound genuinely drops documents.
- **Slow work does not block the event loop.** `transcribe()` and `extract()` are
  synchronous and slow; both go through `run_in_threadpool`. Transcription
  additionally sits behind a semaphore, because faster-whisper holds one model in
  memory and is not safe to call from several threads at once. Extraction is
  deliberately *not* serialised — it is network-bound and already paced.

---

## What was added for accuracy

None of the following was required by the brief. Each exists because the accuracy
of a clinical document is not the same thing as the fluency of a model's answer.

### Citation-based grounding

The obvious design is to ask the model how confident it is. That fails precisely
when it matters, because a model that misread something is confident about the
thing it misread.

Instead, every value must arrive with a verbatim quote, and the quote is verified
**in our own Python** against the transcript. That turns an opinion into a fact
we can check. A value whose quote is not in the transcript scores zero regardless
of how sure the model claims to be.

### Word-level audio confidence

`word_timestamps=True` is the single most important flag in the project. Whisper
reports a probability for every individual word; once we know which span a value
came from, we can ask how clearly *those exact words* were heard.

Without it there is only a per-segment average — which, on the sentence
containing the misheard reading described under Problems, is a reassuring **0.90**.

### A gate and three scores per field

| Signal | Source | Meaning |
|---|---|---|
| `evidenceFound` | our code | Does the quoted span actually exist in the transcript? A gate — false means zero. |
| `audioConfidence` | Whisper | How confidently were *those exact words* heard? |
| `modelConfidence` | the LLM | How sure the model says it is. The weakest, because it is self-reported. |
| `contextConfidence` | Whisper | The worst word in a ±3-word window around the quote. |

The combined score is the **weakest** reported signal — a value is only as
trustworthy as its shakiest evidence. `contextConfidence` bites only below 0.25,
where a neighbouring word is not merely unclear but destroyed.

One subtlety that took a bug to find: a signal of exactly `0` means *not
reported*, not *certain it is wrong*. Gemini frequently omits its own confidence,
and treating that as zero zeroed out every well-grounded field on the page.

### A `422` a human can act on

Fields below `EXTRACTION_CONFIDENCE_THRESHOLD` (default 0.6) come back as `422`
detail, shaped like FastAPI's own validation errors so any client that already
renders a form error can render these with no new code:

```json
{
  "loc": ["assessment", "objectiveAssessment", "tests", 1, "right"],
  "msg": "The recording is badly unclear next to this value (5% on a nearby word)…",
  "type": "low_confidence",
  "ctx": { "value": "5", "confidence": 0.05, "audioConfidence": 0.93 }
}
```

**The 422 still carries the full draft.** An error with nothing attached would
leave a clinician knowing something is wrong and with nothing to correct. The
status code says "do not trust this yet"; the body gives them something to fix.

### A transcript cache

Transcripts are cached on disk under a hash of the audio bytes plus the decode
settings, so `medium` is affordable: 184 seconds once, then nothing. It also
makes the test suite and the end-to-end script fast enough to run repeatedly.

### A review interface

The confidence data is not much use if nobody can see it. `/ui/` is a React app
([`frontend/`](frontend/)) served by the API itself from `app/static/` — same
origin, so there is no CORS configuration anywhere. **The built bundle is
committed**, so a clone runs the UI with no `npm install`.

Selecting any field scrolls the transcript to the words it came from,
highlighted, and shows the three signals behind its score. Flagged fields are
amber (heard poorly) or red (never sourced), and a "Needs review" worklist
collects them worst-first. Only those two states get colour, so they cannot be
missed. A section lost to a failed call says so explicitly rather than rendering
as an ordinary blank.

### Tests that try to break it

```bash
pytest                    # 137 passed, 2 skipped
cd frontend && npm test   # 58 checks
```

Python tests run with **no API key, no Whisper and no Mongo** — the model is
stubbed, because what needs proving is not "does Gemini work" but "does the
grounding catch a model that lies". A stub is the only way to test the lying case
deliberately. MongoDB integration tests skip when no server is reachable; mocking
the driver would only prove the mock behaves like the mock.

| Test | What it pins |
|---|---|
| `test_a_perfect_extraction_from_misheard_audio_is_still_flagged` | The whole design in one test: the agent does everything right, the transcript is wrong, and the field is still caught. |
| `test_a_model_cannot_quote_its_way_around_a_hole_in_the_transcript` | The escape hatch described above, closed. |
| `test_an_ordinary_mumble_nearby_does_not_flag_a_good_value` | That the context check does not cry wolf. |
| `test_a_group_that_returned_nothing_is_retried` | The retry gap that cost a whole section. |
| `test_an_unavailable_section_is_a_422_even_when_the_rest_scored_well` | That a high score cannot mask a missing section. |
| `test_the_contract_survives_the_round_trip_byte_for_byte` | Exact-match through save and load. |

[`tests/run_pipeline.py`](tests/run_pipeline.py) is the end-to-end script. It
checks the **serialised JSON**, not the model — exact keys in order, no nulls,
arrays intact, and that re-validating the document reproduces it byte-identically.
Narration goes to stderr and the contract JSON to stdout, so redirecting produces
a clean document.

---

## Configuration

Every setting has a working default except the two secrets. See
[`.env.example`](.env.example).

| Variable | Default | Notes |
|---|---|---|
| `GOOGLE_API_KEY` | — | Free at [aistudio.google.com/apikey](https://aistudio.google.com/apikey) |
| `MONGODB_URI` | `mongodb://localhost:27017` | Or an Atlas `mongodb+srv://` string |
| `WHISPER_MODEL` | `medium` | `small` is 8× faster and mangles clinical terms |
| `WHISPER_DEVICE` | `cpu` | `cuda` switches compute type to float16 automatically |
| `EXTRACTION_MODEL` | `gemini-3.1-flash-lite` | |
| `EXTRACTION_TEMPERATURE` | `0.0` | Do not raise; sampling invents detail |
| `EXTRACTION_MAX_RETRIES` | `2` | Repair passes before giving up |
| `EXTRACTION_REQUESTS_PER_MINUTE` | `4` | Raise on a paid key |
| `EXTRACTION_CONFIDENCE_THRESHOLD` | `0.6` | Below this, a field goes in the 422 |
| `MAX_UPLOAD_MB` | `50` | ~2 hours of 16-bit mono WAV |
| `MONGODB_TIMEOUT_MS` | `5000` | Short on purpose — fail the request, don't hang the worker |

Audio formats: **WAV, MP3, M4A**, plus FLAC, OGG and WebM. faster-whisper decodes
through its bundled PyAV, so no separate ffmpeg install is needed.

---

## Problems

### The one the whole design exists to catch

In the sample recording the clinician states the patient's right knee extension.
Whisper transcribed it as:

> …left knee extension of 20 degrees compared with **knee gig** 5 degrees on the right.

"knee gig" is not a clinical term. Whisper's *segment-level* confidence for that
sentence is a healthy **0.90**, and the extraction agent behaves impeccably: it
reads `5`, quotes the transcript exactly, and reports 95% confidence. Every check
that looks only at the model passes.

But Whisper's **word-level** probability for the word before `5` is **0.05**. The
clinician almost certainly said "negative 5 degrees" — the sign is inverted, and
a −5° extension is a materially different clinical picture from +5°.

The service returns that field at **0.05** with a `422`:

```
The recording is badly unclear next to this value (5% on a nearby word);
the quote itself is clean, so its meaning may not be.
```

**This is the system working as intended, and it is also a real piece of missing
clinical data.** The true value needs a human to listen to the recording. Nothing
in software can recover it.

### Known limitations

**The 0.6 threshold is tuned against one recording.** It cleanly separates the
confidently-heard values from the misheard ones in the sample, but n=1 is not a
calibration. It lives in config precisely because it is a knob, not a law.

**Extraction varies between runs.** Temperature is 0, but Gemini is not
deterministic. Across runs of the same recording, field counts ranged from 29 to
46 and the set of low-confidence fields shifted slightly. The measurements and
their scores were stable; the softer prose fields were not.

**Units are occasionally reported as unsourced.** The model sometimes omits a
citation for `unitName` even though the transcript says "degrees" eight times.
The system correctly refuses to vouch for an uncited value, but the effect is
noise on an obviously-correct word. Retrying failed groups reduced it; a targeted
prompt instruction would likely finish the job.

**Free-tier rate limits shape the runtime.** A full parse takes ~107 seconds
end-to-end with a cached transcript, most of it waiting on the client-side rate
limiter. On a paid key, raise `EXTRACTION_REQUESTS_PER_MINUTE` and it drops
sharply.

**Atlas M0 clusters auto-pause when idle** and cold-start on the next connection,
which can exceed the 5-second `MONGODB_TIMEOUT_MS`. Not a problem in active use;
worth knowing if a request fails after a long gap.

**The partial-failure path is proven by tests, not by a live run.** Total provider
failure was verified live (`502`). Making exactly one of three concurrent calls
fail against the real API is not something I could arrange, so that path rests on
unit and render tests.
