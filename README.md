# First Assessment

Turns a clinician-patient consultation recording into a structured
`FirstAssessment` document, stores it in MongoDB, and serves it over a REST API.

Transcription runs locally with Whisper. Extraction runs through a LangGraph
agent that must quote the transcript for every value it produces, and every
quote is verified in our own code before the value is returned.

**To run it:** `run.bat` on Windows or `./run.sh` on macOS and Linux. That builds
the environment, verifies it, starts the server, and opens the review interface
and API documentation in your browser.

Setup comes first below, both as that one command and as the individual commands
it runs. Straight after it, [What this achieves](#what-this-achieves) maps each
of the six deliverables to the exact function that implements it and the command
that demonstrates it. How the pipeline works and why it is built this way follow
after that.

---

## Quick start

You need **Python 3.14** installed. The scripts handle everything else.

You do **not** need Node, ffmpeg or Docker. The React bundle is committed to
`app/static`, so the interface runs from a clone with no `npm install`, and
faster-whisper decodes audio through its bundled PyAV.

Two values have to come from you, both free:

| Value | Where from |
|---|---|
| `GOOGLE_API_KEY` | https://aistudio.google.com/apikey |
| `MONGODB_URI` | Atlas free tier, or a local `mongod` |

There are two ways in: one command, or the six commands it runs for you. Both
are below. Why the pipeline is built the way it is comes later, under
[How it works](#how-it-works) and [What changed along the way](#what-changed-along-the-way-and-why).

### One command

```bash
run.bat          # Windows
./run.sh         # macOS / Linux
```

That creates the virtual environment, installs the pins, checks the setup, and
starts the server. Run it with no `.env` present and it copies the example, names
the two values to fill in, and stops, rather than starting a server that will
fail on the first upload.

It then prints where to go, and opens all five in your browser as soon as the
server answers:

```
 ============================================================
  Running at  http://localhost:8000/ui/
 ============================================================

  Review interface   http://localhost:8000/ui/
  API explorer       http://localhost:8000/docs
  API reference      http://localhost:8000/redoc
  Health check       http://localhost:8000/health
  Saved assessments  http://localhost:8000/assessments
```

The tabs are opened by [`app/browser.py`](app/browser.py), which waits for
`/health` to answer first. Opening them the moment the script reaches this line
would race uvicorn, which has not bound the port yet, and land every tab on a
connection error while a perfectly healthy service started up behind it.

Two environment variables adjust this:

| Variable | Effect |
|---|---|
| `PORT` | Serve somewhere other than 8000. Every address above follows it, including the ones the preflight prints. |
| `OPEN_BROWSER=0` | Start the server without opening anything. |

```bash
set PORT=8080 && run.bat            # Windows
PORT=8080 ./run.sh                  # macOS / Linux
set OPEN_BROWSER=0 && run.bat       # no tabs
```

Other modes:

```bash
run.bat check    # only verify the setup, start nothing
run.bat warm     # also pre-download the Whisper weights and test the API key
run.bat test     # run the test suite
```

`warm` is worth the wait once: it pulls the ~1.5 GB of Whisper weights up front,
so the first upload is slow only because of CPU, not because of a download.

Safe to re-run. The venv and installed packages are reused, so a second run
reaches the server in about a second.

### Or step by step

`run.bat` is a wrapper around these six commands. Running them yourself does the
same work, and is the thing to fall back on if any single step misbehaves:

```bash
# 1. Create a virtual environment and activate it
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate

# 2. Install the pinned dependencies
pip install -r requirements.txt

# 3. Supply the two values from the table above
cp .env.example .env            # then fill in GOOGLE_API_KEY and MONGODB_URI

# 4. Check all of that before spending time on it
python -m app.doctor            # --warm also pre-downloads Whisper and tests the key

# 5. Start the service
uvicorn app.main:app --reload --port 8000

# 6. Run the tests, in a second terminal
pytest
```

Then open whichever of these you want, since nothing opens by itself on this
route:

| Address | What it is |
|---|---|
| `http://localhost:8000/ui/` | Review interface: upload, review, correct, save |
| `http://localhost:8000/docs` | API explorer, with every endpoint callable in the page |
| `http://localhost:8000/redoc` | The same API as a reference document |
| `http://localhost:8000/health` | Liveness, including whether MongoDB is reachable |
| `http://localhost:8000/assessments` | Everything saved so far, newest first |

Place the consultation recording at `./clinical_assessment.wav`, then:

```bash
python -m tests.run_pipeline clinical_assessment.wav          # in-process
python -m tests.run_pipeline clinical_assessment.wav --http   # needs the server running
python -m tests.run_pipeline clinical_assessment.wav > out.json
```

`--http` uploads to a service that must already be running, in another terminal
or via `run.bat`. Without it the script drives the same stages in process and
needs no server.

**Three things to expect on a first run:**

1. **Whisper downloads itself.** ~1.5 GB of model weights into
   `~/.cache/huggingface/hub`, once. Nothing is committed to this repo.
   `run.bat warm` does this up front instead of mid-upload.
2. **The first transcription takes minutes**, not seconds: roughly 3 minutes of
   CPU for the 105-second sample. The result is cached by content hash, so every
   run after that is instant.
3. **Two tests skip** until you have transcribed something. They exercise the
   cache round-trip and print the command that satisfies them. Expected on a
   fresh clone, not a failure.

The recording is **not** in this repository. The brief supplies it as an input
file and asks for code, tests and this README back, so audio is in `.gitignore`:
a real patient consultation does not belong in a shared repository.

### If something goes wrong

`run.bat check` (or `python -m app.doctor`) diagnoses most of this in one screen
before the server starts. The rest:

| Symptom | Cause and fix |
|---|---|
| `[FAIL] No Python interpreter found` | Install 3.14 from [python.org](https://python.org/downloads/) and tick "Add python.exe to PATH". On Windows a bare `python` may be the Store stub, which is why the script prefers the `py` launcher |
| `No .env existed, so .env.example was copied` | Working as intended: it stopped instead of starting a server that would fail on the first upload. Fill in the two values and run again |
| `address already in use` | Something else holds 8000. `set PORT=8080 && run.bat`, and every printed address follows |
| `/health` reports `"mongo": false` | Wrong `MONGODB_URI`, or an Atlas M0 cluster that auto-paused while idle and is cold-starting past the 5-second `MONGODB_TIMEOUT_MS`. Try again once |
| `429 RESOURCE_EXHAUSTED` from Gemini | The free tier allows 5 requests per minute. The client already paces itself to 4; on a shared key, lower `EXTRACTION_REQUESTS_PER_MINUTE` or wait a minute |
| The first upload seems to hang | Whisper is downloading ~1.5 GB. `run.bat warm` does that up front instead |
| No browser tabs opened | `OPEN_BROWSER=0` is set, or the server took longer than 90 seconds to answer. The addresses are printed in the terminal either way |
| `bad interpreter: ^M` running `./run.sh` | The clone converted line endings. [`.gitattributes`](.gitattributes) pins `.sh` to LF, so re-clone rather than hand-fixing |

### Changing the interface

The built bundle in `app/static` is committed so a clone needs no Node. To
change the UI you do need it:

```bash
cd frontend
npm install
npm run dev      # hot reload on :5173, proxying the API on :8000
npm run build    # rewrites app/static, which is what the service serves
npm test         # 58 logic and render checks
```

`npm run build` must be re-run and the result committed, or the served interface
will not match the source.

---

## What this achieves

**All six deliverables, working end to end against the real recording and a live
MongoDB Atlas cluster.**

| # | Deliverable | Exactly where it lives |
|---|---|---|
| **D1** | FastAPI service, all 4 endpoints working | [`parse_assessment`](app/main.py#L246) · [`create_assessment`](app/main.py#L312) · [`list_assessments`](app/main.py#L328) · [`get_assessment`](app/main.py#L351) · [`health`](app/main.py#L224), all in [`app/main.py`](app/main.py) |
| **D2** | Whisper transcription module, WAV to text | [`transcribe()`](app/transcription.py#L146) in [`app/transcription.py`](app/transcription.py), local and word-level |
| **D3** | LangGraph agent with `FirstAssessment` Pydantic output | [`build_graph()`](app/extraction.py#L631) and [`extract()`](app/extraction.py#L661) in [`app/extraction.py`](app/extraction.py); the anti-hallucination check is [`ground()`](app/extraction.py#L548) |
| **D4** | MongoDB models, connection, save/retrieve | [`get_client()`](app/db.py#L43) · [`save_assessment()`](app/db.py#L140) · [`get_assessment()`](app/db.py#L148) · [`list_assessments()`](app/db.py#L175) in [`app/db.py`](app/db.py); models are [`FirstAssessment`](app/schemas.py#L194) and [`StoredAssessment`](app/schemas.py#L370) |
| **D5** | Test script: run pipeline on provided WAV, print JSON | [`tests/run_pipeline.py`](tests/run_pipeline.py#L450), narration on stderr and the contract document on stdout |
| **D6** | README: setup instructions and design decisions | this file: [setup](#quick-start), then [How it works](#how-it-works) and [What changed along the way](#what-changed-along-the-way-and-why) |

**Seeing all six at once.** Put the recording at `./clinical_assessment.wav` and
run one command:

```bash
python -m tests.run_pipeline clinical_assessment.wav          # or: run.bat test
```

It transcribes with Whisper (**D2**), runs the LangGraph agent (**D3**), checks
the seven-section contract, scores every field, saves to MongoDB and reads it
back (**D4**), and prints the `FirstAssessment` JSON (**D5**). Add `--http` and
the identical work goes through the running service instead, which exercises all
four endpoints and the rejection paths (**D1**).

### A full run, through the live API

```
POST /assessments/parse   422    45 fields extracted, overall 71%, 17 held back
POST /assessments         201    id 6a877bf166a9d4f67a1aee74
GET  /assessments/{id}    200    contract identical: True
GET  /assessments?date=   200    2026-08-20 → 9,  2020-01-01 → 0

bad id → 404   unknown id → 404   bad date → 422   pdf upload → 415
```

Measurements extracted from the recording, every one traced to a quote:

```
knee flexion           L 124   R 130   degrees
knee extension         L 20    R 5     degrees   ← flagged at 0.05, see Problems
hip internal rotation                  degrees   ← named, no numbers returned
hip external rotation                  degrees   ← named, no numbers returned
ankle dorsiflexion     L 4.5   R 12    degrees
```

That 71% is the honest number rather than the flattering one, and it is worth
saying why it moved. An earlier version of the prompt let the model answer with
the value itself in place of a quote, citing `"5"` as its evidence for the value
`5`. That scores beautifully (overall 95%, two fields held back) and it is
worthless: `"5"` occurs twice in this recording, so the audio check silently took
the better of the two and the misheard measurement below came back at **1.00**.
The prompt now requires a span that matches one place and no other. Scores fell
because they are now being earned.

### What is verifiably true about the output

- **The contract is exact.** Seven sections, in order, no extra keys, no renamed
  keys, arrays that stay arrays, strings that are never null, checked against
  the serialised JSON on the wire, not against the model.
- **Nothing is invented.** Across four runs today every `targetDate` came back
  `""`, because the recording states no dates, and `patientAdvice` came back
  empty, because the clinician gave none. Empty is a correct answer here, and
  the system prefers it to a plausible guess. Which of the two goal lists the
  seven goals land in does vary between runs; see Known limitations.
- **Low-confidence values are held back**, per field, with the reason and the
  evidence behind them.
- **202 automated checks pass:** 144 Python tests (2 skip without a cached
  transcript) and 58 frontend checks.
- **WAV, MP3 and M4A** were each transcoded from the same source and produced
  identical transcripts.

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
a verbatim quote from the transcript. `ground()`, plain Python with no model
involved, then checks that the quote actually appears in the transcript and
looks up how confidently Whisper heard those exact words. **The model never gets
a vote on whether its own output is trustworthy.**

**A value that cannot be traced is repaired, then flagged.** Ungrounded fields
send the graph back through `repair`, which re-asks only the offending group and
names the bad quote. After two attempts, whatever is still unverified is reported
rather than quietly kept.

### The transcription module

[`app/transcription.py`](app/transcription.py) is a WAV in, text out module, and
two things make it more than a wrapper around faster-whisper.

**It keeps the probabilities.** [`transcribe()`](app/transcription.py#L146) runs
with `word_timestamps=True` and returns a
[`TranscriptionResult`](app/schemas.py#L416): the full text, timed segments, and
a probability for every individual word. Throwing those away is what makes a
clinical pipeline dangerous, because "forty degrees" and "fourteen degrees" sound
alike and the agent will extract the wrong one with total confidence. Keeping
them is what later lets the service say *this measurement came from a span
Whisper was 5% sure of*.

**It caches to disk.** The cache key is a hash of the audio bytes *plus* the
decode settings, so changing `WHISPER_MODEL` invalidates it rather than silently
serving a stale transcript. Transcription is the slow, deterministic, expensive
stage and prompt-tuning is the fast, iterative one; without the cache every
prompt tweak would cost another three minutes of CPU.

| | |
|---|---|
| Entry point | [`transcribe(audio_path, use_cache=True)`](app/transcription.py#L146) |
| Returns | [`TranscriptionResult`](app/schemas.py#L416): text, segments, per-word confidence |
| Model | `medium` on CPU by default, `int8` (`float16` on CUDA) |
| Cache | `.cache/transcripts/<hash>.json`, plus a readable `.txt` sibling |
| Formats | WAV, MP3, M4A, FLAC, OGG, WebM, decoded through the bundled PyAV, so no separate ffmpeg install |
| Run it alone | `python -m app.transcription clinical_assessment.wav` |

Run alone it also prints the ten least confident words, which is the fastest way
to see what the recording is going to cause trouble with. On the sample that list
is headed by the 5% `knee` described under [Problems](#problems).

The hosted-API backend deliberately raises `NotImplementedError`: this is patient
audio, and shipping it to a third party is a decision for a compliance review
rather than a default in a config file.

### The extraction agent

[`app/extraction.py`](app/extraction.py) is a LangGraph `StateGraph`, compiled by
[`build_graph()`](app/extraction.py#L631) and driven by
[`extract()`](app/extraction.py#L661), which takes a transcript and returns an
[`ExtractionResult`](app/extraction.py#L650) holding a `FirstAssessment` and its
per-field evidence.

| Node | Line | What it does |
|---|---|---|
| `extract_subjective` | [313](app/extraction.py#L313) | Clinical history, chief complaint, subjective findings |
| `extract_objective` | [313](app/extraction.py#L313) | The measured values |
| `extract_plan` | [313](app/extraction.py#L313) | Goals, recommendation, advice |
| `assemble` | [333](app/extraction.py#L333) | **Builds the `FirstAssessment`** |
| `ground` | [350](app/extraction.py#L350) | Verifies every quote against the transcript |
| `repair` | [421](app/extraction.py#L421) | Re-asks only the group that failed or invented |
| *(routing)* | [460](app/extraction.py#L460) | Conditional edge: repair again, or stop |

The three `extract_*` nodes are generated from one factory over
[`GROUP_SPECS`](app/extraction.py#L175) and all fan out from `START` at once.
They are the "three concurrent calls" described above; each is a separate
structured-output call returning its sections plus a list of
[`Citation`](app/extraction.py#L92) objects.

**Where the contract object is produced.** One line, in `assemble`:

```python
assessment = FirstAssessment.model_validate(
    {k: v for k, v in sections.items() if k in SECTIONS}
)
```

Nothing else in the graph constructs it. A section whose call failed simply stays
at its schema default, which keeps the document valid rather than half-formed.

**The state is typed, and its reducers matter.**
[`ExtractionState`](app/extraction.py#L285) is a `TypedDict` whose fields carry
LangGraph reducers, because the three group nodes write concurrently. `sections`,
`citations` and `failures` merge by key rather than appending, which is what lets
a repair *replace* a group's earlier answer instead of stacking a second one on
top of it. Citations are keyed by the group that produced them, so a field is
only ever checked against quotes from the call that filled it.

**Failure is distinguished from silence.** If every group call fails,
[`extract()`](app/extraction.py#L661) raises
[`ExtractionUnavailable`](app/extraction.py#L80) rather than returning an empty
seven-section document, because an empty document is indistinguishable from a
recording in which the clinician said nothing. The API turns that into a `502`,
not a `422`: nothing was wrong with the request.

| | |
|---|---|
| Entry point | [`extract(transcript, transcription)`](app/extraction.py#L661) |
| Returns | [`ExtractionResult`](app/extraction.py#L650): `assessment` + `flags` |
| Contract model | [`FirstAssessment`](app/schemas.py#L194), validated in [`assemble_node`](app/extraction.py#L333) |
| Model | `gemini-3.1-flash-lite`, temperature 0, paced to 4 requests/minute |
| Prompt | [`SYSTEM_PROMPT`](app/extraction.py#L130), plus a per-group instruction |
| Grounding | [`ground()`](app/extraction.py#L548), plain Python, no model involved |

### The schema contract

The brief's three rules are enforced as code, not convention:

| Rule | How it is enforced |
|---|---|
| No extra fields, no renamed keys | `extra="forbid"` on every model; a renamed key arrives as an unknown key and raises |
| Arrays stay arrays, even with one item | Every list defaults to `[]`; a `None` is normalised to `[]`, never dropped |
| String fields are strings, never null | Every leaf is a `CleanStr`, which maps `None` → `""` at the boundary |

Two details worth naming:

- **Field names are declared in camelCase literally**, not generated by
  `alias_generator=to_camel`. With aliases, a plain `model_dump()` silently emits
  snake_case and only `model_dump(by_alias=True)` is correct, and one forgotten
  flag anywhere writes a wrong-shaped document. Declaring camelCase makes the correct
  output the *only* possible output.
- **`recommendation` is singular but holds an array.** That is the frontend's
  spelling. "Fixing" it would be a renamed key, which the brief forbids.

### MongoDB: models, connection, save and retrieve

[`app/db.py`](app/db.py) holds the persistence layer. Nothing in it reaches
inside `assessment`.

**The document is an envelope, and the contract sits untouched inside it.**
[`StoredAssessment`](app/schemas.py#L370) wraps ids, a timestamp, the transcript
and the confidence metadata *around* an untouched
[`FirstAssessment`](app/schemas.py#L194):

```
{ _id, createdAt, audioFilename, transcript, flags, assessment }
                                                    └─ the contract, exactly
```

That separation is the whole point. The brief forbids extra fields inside the
contract, so everything we need to keep alongside it lives outside it, and no
code path in this module touches the sub-document. The exact-match guarantee
therefore does not depend on the database round trip preserving it; it holds
because nothing reaches in.

**Connection.** [`get_client()`](app/db.py#L43) lazily builds one process-wide
`AsyncMongoClient`. Constructing it does not connect: the first operation does,
which is why a wrong URI surfaces as a timeout on the first save rather than at
import. `serverSelectionTimeoutMS` is deliberately short, so an unreachable
cluster fails the request in seconds instead of hanging a worker for the driver's
30-second default. `tz_aware=True` matters more than it looks: without it BSON
returns naive datetimes and comparing one to the aware `createdAt` raises at
runtime. [`ensure_indexes()`](app/db.py#L83) runs once at startup and is
idempotent; [`close()`](app/db.py#L97) drops the client so the next call rebuilds
it.

**The four operations.**

| Operation | Line | Notes |
|---|---|---|
| [`save_assessment()`](app/db.py#L140) | 140 | Inserts and returns the new id as a string |
| [`get_assessment()`](app/db.py#L148) | 148 | A malformed id returns `None`, not an error: "no such assessment" and "that could never be one" both mean 404 |
| [`list_assessments()`](app/db.py#L175) | 175 | Newest first, optional day filter, `skip`/`limit` |
| [`ping()`](app/db.py#L72) | 72 | Never raises, because a health check that 500s is useless |

**Conversion is explicit.** [`to_document()`](app/db.py#L108) drops `id` because
Mongo owns it, and writes `createdAt` as a real datetime rather than a string so
range queries work. It does *not* write the combined confidence score: only the
three raw signals are stored and the score is recomputed on read, because a
derived value in a database goes stale the moment the scoring rule changes.
[`from_document()`](app/db.py#L123) moves `_id` into `id` and validates strictly,
so a document carrying unknown keys raises rather than quietly dropping them.

**The index matches the query exactly.**
[`LIST_INDEX`](app/db.py#L37) is compound and descending on
`(createdAt, _id)`, the same shape as the list query's sort. Including the `_id`
tie-break is what lets `skip`/`limit` paginate without an in-memory sort. The day
filter uses a half-open range [`_day_range()`](app/db.py#L163), because BSON keeps
milliseconds and an inclusive upper bound genuinely drops documents.

### The four endpoints

```
POST /assessments/parse      multipart audio  ->  draft assessment, or 422
POST /assessments            draft            ->  saved, with an id
GET  /assessments/{id}       id               ->  one assessment
GET  /assessments?date=      YYYY-MM-DD       ->  that day's assessments
```

All four speak one envelope: the untouched `FirstAssessment` under `assessment`,
with transcript, confidence and identifiers wrapped around it. A wrapper is
unavoidable: the brief forbids extra fields *inside* the contract and requires
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

The cost is real, 184 seconds versus 24 on CPU, but it is paid **once per
recording**, because transcripts are cached by a hash of the audio bytes plus the
decode settings. A mangled clinical term, by contrast, is paid on every read of
that assessment forever.

### Confidence per conversation → confidence per field

The first design scored the extraction as a whole. That hides exactly the case
that matters: one destroyed measurement inside an otherwise clean assessment
disappears into an average. Scoring every field independently is what makes the
`422` specific enough to act on, and it reshaped the schema into evidence
records rather than a single number.

### Seven extraction nodes → three

The first version had one node per section. That fired seven concurrent calls and
exhausted a free-tier daily quota in three runs. Grouping into subjective /
objective / plan cut it to three calls per extraction with no loss of focus: the
groups match what a clinician reasons about together.

### `gemini-2.5-flash` → `gemini-3.1-flash-lite`

The 2.5 and 3.7 flash tiers allow 5 requests/minute and a few dozen per day on
the free tier, which two full runs exhaust. `flash-lite` has real headroom
(15/min) and handled the citation discipline without difficulty. Client-side
pacing was added alongside it, held one under the limit rather than at it,
because the limiter can pace what we send but not the retries the SDK fires on
its own.

### Scoring the quote → scoring the quote *and its neighbourhood*

A live run exposed something worth recording. Told to "quote the shortest span
that establishes the value", the model quoted `"5 degrees on the right"`, every
word of which Whisper heard at 93% or better, stepping neatly over the 5% word
beside it. **A perfectly well-behaved model had steered around the safety check.**

The fix was a ±3-word window around every quote. That initially produced false
alarms on misheard function words (a mangled "and" between two goals casts doubt
on neither) so `CONTEXT_IGNORED` excludes them. A misheard "negative" before a
measurement still counts.

### Motor → PyMongo's `AsyncMongoClient`

Motor reached end-of-life in May 2026 once PyMongo 4.13 absorbed the async API.
Starting a new project on a driver with no upstream is not defensible. The API is
near-identical; the only visible difference is the import.

### A silent partial failure → `422`, and a total one → `502`

The extraction makes three concurrent calls. When one failed, its sections came
back empty, and an empty section normally means the clinician did not mention
it, which is a *correct* answer. The two were indistinguishable in the finished
document.

Worse, confidence *rose* when a section was lost, because it averages only the
fields that came back. One run returned 90% overall with a third of the
assessment silently missing.

Three changes fixed it:

1. **A group that returned nothing is now retried.** The repair loop originally
   looked only at fields whose quotes did not check out. A failed call produced
   no fields, so it was invisible to that loop and never asked again. A bad quote
   got two more chances; a dropped connection got none.
2. **Surviving failures name the sections they cost**, in `flags.failedSections`,
   by contract section rather than internal group name.
3. **The status code stopped lying.** A partial failure is a `422` carrying a
   `section_unavailable` entry per lost section, *even when every field that did
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
  already reviewed, and gating it would mean a clinician who corrected a misheard
  measurement could not save the correction.
- **`?date=` filters the envelope's `createdAt`, in UTC.** `FirstAssessment` has
  no date of its own; its only dates are goal `targetDate` values, which are
  intentions rather than a record of when anything happened. The range is
  half-open `[midnight, next midnight)`, because BSON keeps milliseconds, so an
  inclusive upper bound genuinely drops documents.
- **Slow work does not block the event loop.** `transcribe()` and `extract()` are
  synchronous and slow; both go through `run_in_threadpool`. Transcription
  additionally sits behind a semaphore, because faster-whisper holds one model in
  memory and is not safe to call from several threads at once. Extraction is
  deliberately *not* serialised: it is network-bound and already paced.
- **The launcher owns the port, and the browser waits for the server.** The
  scripts pass `--port` explicitly rather than letting uvicorn's default decide,
  so the address they print is the address being served: they agreed at 8000 only
  by coincidence before, and any `PORT` override would have made the message a
  lie. [`app/browser.py`](app/browser.py) then polls `/health` before opening
  anything, because the tabs are launched from the line above `uvicorn` and the
  port is not bound yet at that point. It treats a `503` as ready, since a
  service correctly reporting that MongoDB is unreachable is worth looking at.
- **Line endings are pinned per file type.** [`.gitattributes`](.gitattributes)
  forces LF on `.sh` and CRLF on `.bat` rather than leaving it to each machine's
  `core.autocrlf`. A clone on Windows would otherwise hand macOS a `run.sh` with
  CRLF endings, which bash rejects as `bad interpreter: ^M`, a message pointing
  nowhere near the cause.
- **The scripts share their logic through Python, not through two shells.**
  Preflight lives in [`app/doctor.py`](app/doctor.py) and tab-opening in
  [`app/browser.py`](app/browser.py), so `run.bat` and `run.sh` stay thin
  wrappers. Two shell implementations of the same checks drift, and the one that
  drifts is always the one you are not developing on.

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

Without it there is only a per-segment average, which, on the sentence
containing the misheard reading described under Problems, is a reassuring **0.90**.

### A gate and three scores per field

| Signal | Source | Meaning |
|---|---|---|
| `evidenceFound` | our code | Does the quoted span actually exist in the transcript? A gate: false means zero. |
| `audioConfidence` | Whisper | How confidently were *those exact words* heard? |
| `modelConfidence` | the LLM | How sure the model says it is. The weakest, because it is self-reported. |
| `contextConfidence` | Whisper | The worst word in a ±3-word window around the quote. |

The combined score is the **weakest** reported signal: a value is only as
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
  "msg": "The recording is badly unclear next to this value (5% on a nearby word)...",
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
([`frontend/`](frontend/)) served by the API itself from `app/static/`, same
origin, so there is no CORS configuration anywhere. **The built bundle is
committed**, so a clone runs the UI with no `npm install`.
[`vite.config.js`](frontend/vite.config.js) sets `base: "/ui/"` and builds
straight into `app/static`: without that base the asset URLs come out absolute
from `/` and 404 behind the mount point.

Selecting any field scrolls the transcript to the words it came from,
highlighted, and shows the three signals behind its score. Flagged fields are
amber (heard poorly) or red (never sourced), and a "Needs review" worklist
collects them worst-first. Only those two states get colour, so they cannot be
missed. A section lost to a failed call says so explicitly rather than rendering
as an ordinary blank.

### The end-to-end script

[`tests/run_pipeline.py`](tests/run_pipeline.py) is the script the brief asks
for: run the pipeline on the provided recording and print the JSON. It is driven
by hand rather than by pytest, because it needs a real API key, a real database
and three minutes of CPU.

```bash
python -m tests.run_pipeline clinical_assessment.wav            # in process
python -m tests.run_pipeline clinical_assessment.wav --http     # through the API
python -m tests.run_pipeline clinical_assessment.wav > out.json # just the document
python -m tests.run_pipeline --transcript transcript.txt --no-save
```

**Narration goes to stderr, the contract JSON to stdout.** That is what makes the
third form work: `> out.json` captures a clean `FirstAssessment` document while
the progress report still appears in the terminal.

Six stages, each printing what it proved
([`run_direct()`](tests/run_pipeline.py#L159)):

```
1. Transcription   ok  clinical_assessment.wav: 105.5s, 276 words, language 'en'
2. Extraction      ok  model gemini-3.1-flash-lite
3. Contract        ok  exactly the 7 required sections, in order, no nulls, arrays intact
4. Confidence          45 fields with evidence, overall 71%, 17 below 60%
5. MongoDB         ok  saved, read back byte for byte, date filter includes and excludes
6. FirstAssessment     the document, on stdout
```

**What lands on stdout.** The bare contract, nothing else. Real output from the
run above, trimmed only where an array repeats:

```jsonc
{
  "clinicalDetails": {
    "clinicalHistory": "involved in a road traffic accident resulting in a left tibial condle fracture and an avulsion ACL tear. Open reduction and internal fixation was performed",
    "chiefComplaint": "left knee pain, difficulty performing functional activities and difficulty walking along with ankle and back pain",
    "duration": "eight months"
  },
  "subjectiveAssessments": [
    { "testName": "Surgical scar", "conclusion": "healed surgical scar was noted on the medial aspect of the knee" },
    { "testName": "Knee flexion", "conclusion": "restricted and painful knee flexion on over pressure" }
    // ... 5 more
  ],
  "objectiveAssessment": {
    "tests": [
      { "testName": "knee flexion", "unitName": "degrees", "value": "", "left": "124", "right": "130", "comments": "" },
      { "testName": "knee extension", "unitName": "degrees", "value": "", "left": "20", "right": "5", "comments": "knee gig" }
      // ... 3 more
    ]
  },
  "subjectiveGoals": [],
  "objectiveGoals": [
    { "goalName": "restoring knee extension", "goalCategory": "", "unitName": "", "value": "", "targetDate": "" }
    // ... 6 more
  ],
  "recommendation": [
    { "sessionType": "Physiotherapy", "sessionFrequency": "once weekly for four sessions" }
  ],
  "patientAdvice": { "adviceDetails": "" }
}
```

Three things in that document are worth pointing at. Every `targetDate` is `""`,
because the recording states no dates and a guessed one would be an invented
clinical date. `patientAdvice` is empty for the same reason: the clinician gave
none, and empty is the correct answer. And `tests[1]` is the misheard measurement
described under [Problems](#problems): the value `"5"` is returned, but the
`comments` field carries the mangled `"knee gig"` beside it and the field comes
back flagged at 0.05 rather than silently trusted.

The real file is 3.4 KB and parses as valid JSON; `python -m tests.run_pipeline
clinical_assessment.wav > out.json` is the form that produces it cleanly.

**Stage 3 checks the serialised JSON, not the model.**
[`check_contract()`](tests/run_pipeline.py#L84) verifies exact keys in order, no
nulls where a string belongs, arrays that stayed arrays, and that re-validating
the document reproduces it byte for byte. Checking the model object instead would
prove only that Pydantic agrees with itself.

**`--http` runs the identical work through the live service**
([`run_http()`](tests/run_pipeline.py#L306)), which is what proves the four
endpoints, the 422 and the persistence layer agree with each other rather than
merely working alone. It adds a stage that asserts the rejection paths: a PDF is
refused with 415, an empty file with 400, a malformed and an unknown id with 404,
and an unparseable date with 422.

**Exit code 0 if every stage passed, 1 if one failed.** Low confidence is
deliberately *not* a failure, because a 422 is the service working correctly. An
extraction that produced nothing at all is, because that is indistinguishable
from success.

### Tests that try to break it

```bash
pytest                    # 142 passed, 2 skipped; 144 once a transcript is cached
run.bat test              # the same, without activating the venv first
cd frontend && npm test   # 58 checks
```

| File | Tests | What it covers |
|---|---|---|
| [`tests/test_schema.py`](tests/test_schema.py) | 50 | The exact-match contract: key order, no nulls, arrays that stay arrays |
| [`tests/test_api.py`](tests/test_api.py) | 39 | All four endpoints, the shape of the 422, every rejection path |
| [`tests/test_extraction.py`](tests/test_extraction.py) | 27 | Grounding and the repair loop, with the model stubbed |
| [`tests/test_db.py`](tests/test_db.py) | 17 | Save, retrieve, and the date filter |
| [`tests/test_transcription.py`](tests/test_transcription.py) | 11 | Confidence conversion and the cache. The 2 that skip need a cached transcript |
| [`tests/run_pipeline.py`](tests/run_pipeline.py) | n/a | The end-to-end script, run by hand rather than by pytest |

Collection settings are in [`pytest.ini`](pytest.ini); the frontend checks are
[`check-logic.mjs`](frontend/scripts/check-logic.mjs) and
[`check-render.mjs`](frontend/scripts/check-render.mjs), run by
[`npm test`](frontend/package.json).

Python tests run with **no API key, no Whisper and no Mongo**. The model is
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

---

## Configuration

Every setting has a working default except the two secrets. The definitions
live in [`app/config.py`](app/config.py) as a `pydantic-settings` model, so an
out-of-range or misspelled value fails at startup rather than halfway through a
request. [`.env.example`](.env.example) is the file to copy.

`run.bat check` (or `python -m app.doctor`) validates this configuration before
the server starts: an unset key, an unreachable database or a half-finished
install is reported as one line saying what to do, rather than as a traceback
after Whisper has already spent three minutes transcribing.

| Variable | Default | Notes |
|---|---|---|
| `GOOGLE_API_KEY` | none | Free at [aistudio.google.com/apikey](https://aistudio.google.com/apikey) |
| `MONGODB_URI` | `mongodb://localhost:27017` | Or an Atlas `mongodb+srv://` string |
| `WHISPER_MODEL` | `medium` | `small` is 8× faster and mangles clinical terms |
| `WHISPER_DEVICE` | `cpu` | `cuda` switches compute type to float16 automatically |
| `EXTRACTION_MODEL` | `gemini-3.1-flash-lite` | |
| `EXTRACTION_TEMPERATURE` | `0.0` | Do not raise; sampling invents detail |
| `EXTRACTION_MAX_RETRIES` | `2` | Repair passes before giving up |
| `EXTRACTION_REQUESTS_PER_MINUTE` | `4` | Raise on a paid key |
| `EXTRACTION_CONFIDENCE_THRESHOLD` | `0.6` | Below this, a field goes in the 422 |
| `MAX_UPLOAD_MB` | `50` | ~2 hours of 16-bit mono WAV |
| `MONGODB_TIMEOUT_MS` | `5000` | Short on purpose: fail the request, don't hang the worker |

Audio formats: **WAV, MP3, M4A**, plus FLAC, OGG and WebM. faster-whisper decodes
through its bundled PyAV, so no separate ffmpeg install is needed.

---

## Problems

### The one the whole design exists to catch

In the sample recording the clinician states the patient's right knee extension.
Whisper transcribed it as:

> ...left knee extension of 20 degrees compared with **knee gig** 5 degrees on the right.

"knee gig" is not a clinical term. Whisper's *segment-level* confidence for that
sentence is a healthy **0.90**, and the extraction agent behaves impeccably: it
reads `5`, quotes the transcript exactly, and reports 95% confidence. Every check
that looks only at the model passes.

But Whisper's **word-level** probability for the word before `5` is **0.05**. The
clinician almost certainly said "negative 5 degrees", so the sign is inverted and
a -5° extension is a materially different clinical picture from +5°.

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
46 and the set of low-confidence fields shifted. The five measurements and the
seven goals were stable, as was every empty field; what moved was which of the
two goal lists the goals landed in, whether the hip rotations came back with
numbers or only names, and which prose fields fell below the bar.

**Fields that share a value share a citation.** Matching is by value, so the five
`unitName` fields, all holding "degrees", all resolve to whichever citation the
model raised for "degrees" and inherit that one span's audio score. Grouping
citations by the call that produced them fixed the cross-section case; telling
two fields apart *within* one call needs the model to say which field each
citation belongs to, which is the array-index problem this design deliberately
keeps away from the model.

**A longer quote scores lower, by construction.** `audioConfidence` is the
weakest word inside the quoted span, so a span that pins down its value also
drags in more words that Whisper may have heard poorly. Requiring spans that
locate uniquely therefore lowered scores across the board: on this recording
"degrees" in "knee flexion of 124 degrees" was heard at 52%, and every field
citing that span now sits below the 0.6 bar. That is a real signal rather than
noise, but it is the reason the overall number reads 71% rather than 95%.

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
