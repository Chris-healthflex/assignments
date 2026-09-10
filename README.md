# Clinical Assessment Pipeline

A recorded physiotherapy session in, a `FirstAssessment` JSON document out —
with a hard guarantee that no clinical value reaches the output unless it was
actually spoken in the recording.

## The problem, and the idea that solves it

Transcribe a consultation, hand it to a language model, and ask it to fill in a
clinical form, and it will fill in the form. That is precisely the danger. A
model asked for a range-of-motion measurement will produce a plausible number
whether or not one was taken, and nothing in the output distinguishes a
transcribed `45 degrees` from an invented one. In a clinical record, that is not
a rough edge — it is the whole risk.

**So this pipeline never asks the model for the production schema.**

The model fills an *internal* shape in which every leaf is an
`ExtractedValue{value, evidence}` — the clinical datum, plus the verbatim span
of transcript that supports it. A **pure function** maps that onto
`FirstAssessment`, and a **deterministic verifier** — no model involvement —
checks every quote against the transcript before anything is emitted.

Two properties fall out, and both are requirements rather than embellishments:

- **Exact key naming stops depending on the model.** Keys come from `schema.py`,
  so a renamed or extra key is structurally impossible rather than
  prompt-dependent.
- **Hallucination becomes checkable.** A model that must quote its source can be
  caught quoting something that isn't there. A model that merely returns values
  cannot be. Anything failing verification is blanked to `""`, never passed
  through.

### It demonstrably works

On the supplied recording, the clinician states that the right knee extends to
*"negative five degrees"*. The first extraction left one field ungrounded. The
repair cycle re-asked for that field alone, and the second attempt was clean.
The final artifact records:

```json
"right": "negative five degrees"
```

Words, not `-5` — because no digit appears anywhere in that field's evidence,
and the digit rule rejects a numeral that its own quote cannot support. The
constraint is visible in the output, not just in the prose.

---

## At a glance

| | |
|---|---|
| Tests | **179 unit + 2 integration**, all passing |
| Schema conformance | **21/21** key paths, checked field-by-field against the brief |
| Latest run | confidence **1.00** after **1 repair round** |
| API | **4 endpoints**, FastAPI, async throughout |
| External binaries | **none** — no `ffmpeg`, audio decoded with the standard library |
| Python | 3.11+ |
| Setup | **one command** — `python bootstrap.py` |

## Deliverables

| | Deliverable | Where | Rationale in |
|---|---|---|---|
| D1 | REST API, four endpoints | `backend/src/clinical_assessment/api.py` | *The two 422s*, *Error mapping* |
| D2 | Transcription | `transcription.py` | *No ffmpeg* |
| D3 | Extraction agent | `agent.py`, `grounding.py`, `mapping.py` | *Grounding is arithmetic*, *Why LangGraph* |
| D4 | Persistence | `storage.py` | *`AsyncMongoClient`, not Motor* |
| D5 | CLI script | `backend/scripts/run_pipeline.py` | *Run* |
| D6 | Documentation | this file | — |

---

## Quickstart

```bash
python bootstrap.py
```

One command, no arguments. It creates the virtual environment, installs the
pinned dependencies and the package, starts MongoDB if Docker is available,
seeds `backend/.env`, and finishes by running the test suite — so setup ends in
proof rather than a claim:

```
[1/5] Virtual environment
      reusing .venv
[2/5] Dependencies
      dependencies installed
[3/5] MongoDB
      reusing container stance-mongo
[4/5] Environment file
      created backend/.env from the example
[5/5] Verifying with the test suite
179 passed, 2 deselected, 1 warning in 5.29s

Setup complete.
```

Re-running is safe — an existing environment, container or `.env` is reused,
never clobbered. It also degrades rather than failing: with no Docker it warns,
skips MongoDB and carries on, because only EP2–EP4 need a database.

**The one thing it cannot do for you** is supply the API key. Put it in
`backend/.env`, or export it — an exported variable wins:

```bash
ANTHROPIC_API_KEY=sk-ant-...
```

<details>
<summary>Manual setup, if you would rather not run a script</summary>

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows;  source .venv/bin/activate elsewhere
pip install -r backend/requirements.txt
pip install -e backend

export ANTHROPIC_API_KEY=sk-ant-...                        # required for extraction
docker run -d -p 27017:27017 --name stance-mongo mongo:7   # required for storage
```

</details>

### Configuration

Every setting has a working default; all are environment variables.

| Variable | Default | Meaning |
|---|---|---|
| `ANTHROPIC_API_KEY` | — | Read by the SDK. Never logged, never in source |
| `WHISPER_MODEL_SIZE` | `medium` | Whisper checkpoint. `small` is ~10× faster but loses clinical detail — see *Known limitations* |
| `CONFIDENCE_THRESHOLD` | `0.70` | Grounded ratio below which parsing returns 422 |
| `MONGODB_URI` | `mongodb://localhost:27017` | Assessment store |
| `ANTHROPIC_MODEL` | `claude-sonnet-5` | Extraction model |

### Run the pipeline (D5)

Assessment JSON to stdout, progress to stderr:

```bash
python backend/scripts/run_pipeline.py backend/assets/clinical_assessment.wav > assessment.json
```

On PowerShell, `>` prepends a UTF-8 BOM, which strict JSON readers reject. Use
`... | Out-File -Encoding utf8NoBOM assessment.json` there instead.

### Run the API (D1)

```bash
uvicorn clinical_assessment.api:app --reload
```

| # | Route | Behaviour |
|---|---|---|
| EP1 | `POST /assessments/parse` | multipart WAV → `{assessment, extraction}`; no persistence |
| EP2 | `POST /assessments` | persist a parsed result → `{id}` |
| EP3 | `GET /assessments/{id}` | retrieve by id |
| EP4 | `GET /assessments?from=&to=` | list, filterable by ISO-8601 date |

```bash
curl -F "file=@backend/assets/clinical_assessment.wav" http://localhost:8000/assessments/parse
```

### Run the tests

From `backend/`, which is where `pyproject.toml` lives:

```bash
cd backend
pytest                      # unit suite; no network, no database, no model calls
pytest -m integration       # adds live-MongoDB round trips; skipped without MONGODB_URI
```

---

## Results

One run on the supplied recording. Whisper `medium`, CPU, `claude-sonnet-5`:

| Stage | Measurement |
|---|---|
| Input | 4,654,720 samples @ 44.1 kHz, mono, 16-bit = **105.5 s** |
| Transcription | 656.6 s → 1,835 characters |
| Extraction | two calls, both HTTP 200, **1 repair round** |
| Confidence | **1.00** after the repair round |
| Conformance | validates against `FirstAssessment`; no extras, no nulls, all strings |

### The repair cycle earning its place

The headline result is not the 1.00 — it is *how* the run reached it.

```
Requesting extraction from claude-sonnet-5 ... HTTP 200
Repair round 1 for 1 ungrounded field(s)
Requesting extraction from claude-sonnet-5 ... HTTP 200
Confidence 1.00 after 1 repair round(s)
```

The verifier rejected a field, `repair` re-asked for **only that field** naming
the quote that could not be found, and the second attempt was clean. A linear
chain would have had to either emit the unsupported value or blank it. The cycle
recovered it instead — which is the entire argument for using a graph here
rather than a straight pipeline.

### What the verification rules demonstrably did

- **`duration` is `"eight months"`, not `"8 months"`.** Converting the words to a
  numeral would have failed the digit rule, because the digits never appear in
  the evidence span.
- **Every `targetDate` is `""`.** No dates were spoken. Dates are the field most
  likely to attract a plausible-sounding guess, and none was invented.
- **`patientAdvice.adviceDetails` is `""`.** No home advice was given in the
  session, so none was written down.

### What confidence 1.00 means — and what it does not

Recounted directly from `assessment.json`:

| Measure | Value |
|---|---|
| Total leaves | 74 |
| Populated, and grounded | 39 |
| Correctly blank | 35 |
| **Confidence** (grounded ÷ total) | 74/74 = **1.00** |
| **Completeness** (populated ÷ total) | 39/74 = **53%** |

These measure different things, and only the first is the gate. **1.00 means
nothing unsupported was emitted — not that the form is complete.** A field the
model correctly left blank counts as grounded, so a sparse session scores well
by design. Completeness is the honest companion: it says how much of the form
this particular session actually covered. Of the 39 fields carrying a value, all
39 passed verification — one of them only after the repair round.

The blanks cluster where the session had nothing to say: 24 across
`objectiveGoals` (six goals were named, but no targets, units or dates were
stated), 10 across the six `objectiveAssessment.tests`, and
`patientAdvice.adviceDetails`.

---

## How it works

```
WAV ─→ transcript ─→ extract ─→ verify ─┬─ ungrounded, attempts left ─→ repair ─┐
      (Whisper,      (evidence-  (pure) │                                       │
       local)         carrying)         └─ otherwise ─→ finalize ─→ FirstAssessment
                                         ▲                                      │
                                         └──────────────────────────────────────┘
```

### Grounding is arithmetic, not self-report

`grounding.py` verifies each field with no model involvement:

- the evidence must appear in the transcript, after normalising case,
  punctuation and whitespace;
- **every digit run in a value must appear in that field's own evidence.**

The digit rule is the specific defence against the worst failure mode: a real
quote paired with a fabricated number. Digit runs are compared *whole*, so a
value of `"45"` is rejected on evidence reading `"145 degrees"`.

Confidence is then `grounded ÷ total` — arithmetic over field verdicts, not the
model's opinion of itself. A field the model correctly left blank counts as
grounded: "not stated in this session" is a correct answer, and scoring it as a
failure would 422 every sparse session. The cost of that choice is that the
ratio measures *"nothing unsupported was emitted"* rather than *"the form is
complete"* — which is why *Results* reports completeness beside it.

### Why LangGraph rather than a linear chain

`verify` is deterministic; when it rejects fields, `repair` re-asks for only
those fields, naming the quotes that were not found. Without that loop this
would be a straight line and a graph library would be decoration. The repair
budget is capped at 2, and the cap is tested with an adversarial always-fails
stub asserting a bounded call count — an unbounded cycle in a paid API loop is a
production incident, not a bug.

### No ffmpeg

`ffmpeg` is not installed on the target machine, and Whisper accepts a float32
waveform directly. So the audio path is the standard library: `wave` to decode,
`numpy` to downmix and resample to 16 kHz by linear interpolation. This also
keeps the audio functions unit-testable without importing torch — the suite
asserts that neither `torch` nor `whisper` is imported while it runs.

`scipy.signal.resample_poly` was rejected as a whole dependency for one call;
Whisper mel-spectrograms its input, so the interpolation difference is inaudible
to it.

### `AsyncMongoClient`, not Motor

Motor reached deprecation in May 2026. `pymongo.AsyncMongoClient` is the
supported async driver and needs no third-party wrapper.

### Confidence data lives beside the assessment, never inside it

"No extra fields" constrains the `FirstAssessment` object. The envelope around
it is ours, so both the API response and the stored document are
`{assessment: {...}, extraction: {overallConfidence, lowConfidenceFields, ...}}`.
The `assessment` value alone is what the frontend consumes.

### camelCase attributes, not `Field(alias=...)`

With aliases, correct output depends on every caller remembering
`model_dump(by_alias=True)`; one forgotten flag silently emits snake_case. The
attribute names *are* the wire contract, so they are written literally. This is
also why ruff's `N` (pep8-naming) ruleset is deliberately not enabled — N815
would fight the requirement rather than help.

### The two 422s

FastAPI already returns 422 for its own request validation, so the confidence
gate uses a distinguishable body:

| Source | `detail` |
|---|---|
| FastAPI validation | a **list** of validation errors |
| Confidence gate | an **object** with `"error": "low_confidence"`, the ratio, the threshold, and a field-level list |

```json
{"detail": {"error": "low_confidence", "overallConfidence": 0.55, "threshold": 0.7,
  "lowConfidenceFields": [{"field": "objectiveAssessment.tests[0].value",
                           "reason": "value contains digits that its evidence does not"}]}}
```

### Error mapping

| Condition | Status |
|---|---|
| Undecodable upload | 400 |
| Unknown or malformed assessment id | 404 |
| Confidence below threshold | 422 + field-level detail |
| Transcription or extraction failure | 500 |
| Store unreachable | 503 |

500 and 503 bodies are deliberately generic: the underlying messages can carry a
temporary file path or a connection string. Two tests assert nothing leaks.

### "3 endpoints" vs "4 endpoints"

The brief's step 04 header says three endpoints while the same step lists EP1–EP4
and deliverable D1 says "all 4 endpoints working". **Four are built**, since D1
is the explicit deliverables list.

---

## Layout

```
bootstrap.py                  one-command setup; verifies itself with the suite
assessment.json               the real pipeline output for that recording
backend/
├── pyproject.toml            build, ruff, black and pytest config
├── requirements.txt          pinned dependencies
├── .env.example              placeholder settings; copy to .env
├── assets/                   the supplied recording lives here (gitignored:
│                             input, not source - place your copy here)
├── scripts/run_pipeline.py   CLI entry point (D5)
├── tests/                    one module per source module
└── src/clinical_assessment/
    ├── schema.py             the production contract; frozen first
    ├── errors.py             one exception hierarchy for the whole pipeline
    ├── config.py             environment settings
    ├── transcription.py      WAV → text, no ffmpeg
    ├── extraction_models.py  evidence-carrying internal shape
    ├── prompts.py            extraction and repair prompts
    ├── llm.py                schema-constrained Anthropic call
    ├── grounding.py          verification; pure
    ├── mapping.py            internal → production schema; pure
    ├── agent.py              LangGraph graph with the repair cycle
    ├── storage.py            AsyncMongoClient repository
    └── api.py                FastAPI, four endpoints
```

The src layout sits inside `backend/`: `pyproject.toml` resolves both
`packages.find where = ["src"]` and pytest's `pythonpath = ["src"]` relative to
its own directory, so sitting beside `src/` is what makes them work.

**The conformance test is the contract's guard.** `backend/tests/test_schema.py`
owns a frozen literal set of all 21 production key paths and asserts set
equality against a dumped model — catching a missing key *and* an extra key in
one assertion. The frozen set lives in the test rather than in the package on
purpose, so production code cannot drift the contract into agreement with
itself.

---

## Known limitations

- **No speaker diarization.** Whisper returns unlabelled text, so the prompt
  infers role from context (the chief complaint is the patient's, the
  recommendation the clinician's). Adding `pyannote` would mean a second model,
  a HuggingFace token and a large dependency for marginal gain on a two-party
  session.
- **The digit rule is strict.** A spoken "three weeks" extracted as `"3 weeks"`
  fails verification, because `3` never appears in the evidence. That is correct
  under "never state a number that was not spoken", but it is the most likely
  source of a false low-confidence flag. The repair round exists to recover it —
  and on the real run, it did.
- **Transcription quality bounds everything.** A mis-heard clinical term cannot
  be recovered downstream, so this was measured rather than assumed. Both
  checkpoints were run over the same recording:

  | `small` heard | `medium` heard |
  |---|---|
  | `left-to-be-all condylo fracture` | `left tibial condylo fracture` |
  | `ankle doser flexion` | `ankle dorsiflexion` |
  | `knee gig 5°` | `negative five degrees` |

  The third row is why the default is `medium`: `small` did not merely garble a
  word, it destroyed a **measurement**. The cost is real — 61.6 s versus 656.6 s
  on CPU, roughly 10× — but for a batch pipeline over a 105-second recording
  that is not the binding constraint. `initial_prompt` seeds physiotherapy
  vocabulary either way.

  `medium` is still not perfect: it writes *condylo* where the clinician said
  *condyle*.
- **Whisper's prompt window is ~224 tokens**, so the vocabulary seed is kept
  short deliberately — an over-long prompt crowds out audio context.

## Possible extensions

Deliberately **not** implemented — noted rather than built:

- number-word normalisation ("three" ↔ "3") to soften the digit rule
- `word_timestamps` to anchor evidence by character offset instead of substring
- prompt caching on the system prompt, which is identical across extract and repair
- server-side refusal fallbacks on the Anthropic call
- per-section confidence instead of one global ratio
- pagination on the listing endpoint; a `lifespan` hook to create indexes at startup
- a LangGraph checkpointer so a failed run resumes rather than restarts
