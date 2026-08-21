# Voice/Note → Structured Clinical Assessment Form Filler

Turns a clinician-patient audio session into a structured `FirstAssessment`
JSON document, matching the exact format Stance Health's clinician frontend
consumes — and, for every value it fills in, records the moment in the
recording it came from.

## Stack

- **FastAPI** — HTTP API
- **Groq Whisper API** (`whisper-large-v3`) — time-coded audio transcription
- **Groq `openai/gpt-oss-120b` via LangGraph** — self-correcting 4-node
  extraction pipeline (extract → validate → refine ↺ → check confidence)
- **Pydantic v2** — strict `FirstAssessment` schema
- **MongoDB (Motor, async)** — persistence
- **React + TypeScript + Tailwind (Vite)** — demo frontend (not a graded
  deliverable; the JSON output is what the assignment actually requires)

## Why these choices

- **Groq instead of OpenAI**: the assignment brief names "OpenAI Whisper" and
  "LangChain/LangGraph" for extraction. Groq serves the same open-source
  Whisper model (`whisper-large-v3`) over an OpenAI-compatible API, so
  transcription quality is unchanged — the substitution is the inference
  provider, not the model family. For extraction, Groq hosts
  `openai/gpt-oss-120b` with tool-calling, which
  `ChatGroq(...).with_structured_output(...)` uses the same way `ChatOpenAI`
  would. The swap was made because a Groq key was available and OpenAI's
  wasn't; both `transcription.py` and `extraction_graph.py` isolate the
  provider behind a small interface (`Groq` client / `StructuredLLM`
  protocol), so switching back to OpenAI is a localized change, not a rewrite.

- **Never hallucinate — and prove it.** Telling a model "don't invent values"
  is a hope, not a guarantee. So the pipeline makes the model cite its
  sources: Whisper returns a *time-coded* transcript, the transcript is fed to
  the extractor as numbered segments, and the extractor must return, for every
  field it fills, the segment ids that justify it. Any populated field with no
  citation is an **ungrounded field** — a hallucination candidate. The graph
  sends those back to the model to fix; whatever survives is surfaced to the
  reviewer as `⚠ unverified` rather than quietly shipped.

- **A self-correcting 4-node graph over a single LLM call.** This is where
  LangGraph earns its place — a straight-line chain can't loop:

  ```
  extract ──► validate ──► refine ──┐   (up to MAX_REFINEMENTS times)
                  ▲                 │
                  └─────────────────┘
                  │
                  └──► check_confidence ──► END
  ```

  `validate` is deterministic Python, not a prompt: it checks the model didn't
  invent section names, didn't cite segment ids that don't exist, didn't stuff
  placeholder text ("N/A", "not mentioned") into fields it should have left
  empty, and didn't fill fields it can't ground in the transcript. If it finds
  problems, the router sends the state to `refine` with the specific
  complaints, and back through `validate`. Keeping validation as code makes it
  testable without an LLM and impossible for the model to talk its way past.

- **Confidence as its own node.** `check_confidence` turns "too many uncertain
  sections" into a decision the API can act on. Keeping the threshold out of
  the prompt makes it testable and tunable (`CONFIDENCE_FLAG_THRESHOLD`)
  without touching the model.

- **Chunking and caching in the transcription layer.** Whisper's upload limit
  is finite, so long recordings are split on WAV frame boundaries and each
  chunk's timestamps are shifted back into whole-file time — the segment ids
  the extractor cites stay meaningful across the whole session. Transcripts
  are cached by SHA-256 of the audio bytes, so re-running the same file during
  review costs nothing. Retries use exponential backoff and fire *only* on
  genuinely retryable errors (rate limits, connection failures, 5xx) — a 400
  fails immediately instead of being hammered four times.

- **MongoDB Atlas over local Mongo**: no local `mongod`/Docker install needed.
- **`mongomock-motor` for tests**: full async Motor test coverage without
  touching a real database.

## Project layout

```
app/
  main.py                       FastAPI app, lifespan-managed Mongo client
  config.py                     env-based settings
  observability.py              JSON logging + request-id middleware
  schemas/first_assessment.py   the FirstAssessment Pydantic models
  services/transcription.py     Groq Whisper wrapper: time-coded transcript,
                                 chunking, content-addressed cache, retries
  services/extraction_graph.py  4-node self-correcting LangGraph pipeline
  db/mongo.py                   Motor-backed repository
  api/assessments.py            the 4 REST endpoints
scripts/run_pipeline.py         CLI: WAV in, FirstAssessment JSON out
samples/                        real output from the provided WAV (see below)
tests/                          pytest suite (51 tests)
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# then fill in GROQ_API_KEY and MONGO_URI in .env
```

Get a free `GROQ_API_KEY` at https://console.groq.com/keys.

`MONGO_URI` should point at a MongoDB Atlas cluster (free M0 tier is enough):
create a cluster at https://cloud.mongodb.com, add a database user, allow
your IP, and copy the `mongodb+srv://...` connection string.

## Running the API

```bash
uvicorn app.main:app --reload
```

- `POST /assessments/parse` — multipart WAV upload → `FirstAssessment` JSON
  (or `422` with `low_confidence_sections` if extraction confidence is low).
- `POST /assessments` — save a `FirstAssessment` JSON body → `{"id": ...}`
- `GET /assessments/{id}` — fetch a saved assessment
- `GET /assessments?date_from=...&date_to=...` — list, optionally filtered by
  `createdAt`

Interactive docs at `http://localhost:8000/docs`.

### The `include_debug` escape hatch

`POST /assessments/parse?include_debug=true` returns the full pipeline record
instead of the bare assessment, and skips the 422 so a human can review and
complete flagged sections rather than being blocked outright:

```jsonc
{
  "assessment":  { /* the graded FirstAssessment — shape unchanged */ },
  "transcript":  "full text",
  "segments":    [{ "id": 0, "start": 0.0, "end": 4.9, "text": "..." }],
  "evidence":    [{ "field": "clinicalDetails.chiefComplaint",
                    "segmentIds": [0], "quote": "..." }],
  "ungrounded_fields":  ["objectiveGoals[0].targetValue"],
  "validation_issues":  [],
  "attempts":    1,
  "is_low_confidence": false,
  "low_confidence_sections": [],
  "confidence":  1.0
}
```

**The graded/default response shape is unchanged either way** — provenance
data is deliberately kept out of the `FirstAssessment` document, so what gets
saved to Mongo is exactly what the brief specifies. `evidence` fields use
dotted paths (`subjectiveAssessments[0].testName`); a citation on a parent
path covers its children.

### Observability

Every request gets an `X-Request-ID` (echoed from the client if supplied,
generated otherwise) that flows through a `ContextVar` into every log line
for that request. Logs are structured JSON — method, path, status,
`duration_ms`, request id — so they can be shipped to a log aggregator
without a regex parser. The header is CORS-exposed so the frontend can
surface it when reporting a failure.

## Sample output

`samples/` holds the real, unedited output of running this pipeline against
the WAV provided with the assignment:

- `clinical_assessment.transcript.txt` — the Whisper transcript
- `clinical_assessment.output.json` — the extracted `FirstAssessment`

That run produced 18 timestamped segments across 105 seconds of audio, with
every populated field grounded in a cited segment and no refinement loop
needed.

## Running the frontend (demo UI)

```bash
cd frontend
npm install
npm run dev
```

Opens at `http://localhost:5173`. It expects the API at
`http://localhost:8000` by default — copy `frontend/.env.example` to
`frontend/.env` and change `VITE_API_BASE_URL` if the API runs elsewhere.
CORS is enabled on the API for `http://localhost:5173` (see `app/main.py`).

Two views: **New Assessment** (drop/pick a WAV → review the extracted
`FirstAssessment` against the recording → save it) and **Saved Assessments**
(browse and filter everything previously saved).

### The verification loop

The interesting part of the UI is not the form, it's the audit trail. After a
parse you get the recording, its time-coded transcript, and the extracted
assessment on one screen. Every extracted value carries a `⏱` citation
button; clicking it highlights the exact transcript segments the model cited
and seeks the audio there. Anything the model filled in but couldn't ground is
marked `⚠ unverified` and its input is tinted amber, with a banner counting
how many need a look. Checking a suspicious value takes a click instead of a
re-listen.

Everything is editable before saving, so a reviewer corrects rather than
re-runs.

### Other UI details

- **Command palette** (`⌘K` / `Ctrl+K`) — pages register the actions they can
  currently perform and withdraw them on unmount, so the palette never offers
  "Save to MongoDB" on a page with nothing to save.
- **Dark mode** — follows the OS on first visit, then remembers the explicit
  choice. Class-based via Tailwind v4's `@custom-variant`.
- **History filtering** — date bounds are pushed to Mongo (the collection
  grows unbounded); free-text search filters client-side, since it only
  refines what's already on screen.
- **JSON export** on both the review screen and any saved assessment.
- Respects `prefers-reduced-motion`; unsaved work warns before unload.

### Frontend architecture

```
frontend/src/
  main.tsx                  Router, ErrorBoundary, ToastProvider, CommandProvider
  App.tsx                   layout shell (header/nav/theme toggle) + <Outlet/>
  pages/
    UploadPage.tsx           "/"           — upload, verify, edit, save
    HistoryPage.tsx          "/history"    — saved assessments, filtered
    AssessmentDetailPage.tsx "/history/:id"— saved assessment detail
  hooks/
    useParseAssessment.ts    parse mutation: status/result/error
    useSaveAssessment.ts     save mutation: status/savedId, double-submit-safe
    useAssessments.ts        list query: data/loading/error/refetch
    useAssessment.ts         by-id query: data/loading/error
    useTheme.ts              persisted light/dark, OS default on first visit
  schemas.ts                 Zod schemas mirroring the FirstAssessment
                              Pydantic models — single source of truth for
                              src/types.ts (derived via z.infer)
  api.ts                     fetch wrappers; every response is validated
                              against schemas.ts before the caller sees it
  components/
    AssessmentView.tsx        read-only or editable FirstAssessment, evidence-aware
    EvidencePanel.tsx         audio player + clickable time-coded transcript
    CommandPalette.tsx        ⌘K palette + the registry pages contribute to
    ConfidenceBadge.tsx       coverage score + formula explainer
    ErrorBoundary.tsx         catches render errors, offers a way back
    ui/                       Button, Card, Badge, Toast — small design system
                               used everywhere instead of ad hoc Tailwind strings
```

Real client-side routes (not tab state) — `/history/:id` is a shareable,
refresh-safe URL; browser back/forward works. API responses are validated at
the boundary with Zod (`api.ts`) rather than trusted via a type assertion, so
a backend contract change surfaces as a clear `ApiShapeError`, not a silent
bad render. Data-fetching lives in hooks, not components, so pages stay
mostly presentational. `AssessmentView` distributes evidence through React
context rather than prop-drilling it through every one of ~30 fields.

## Running the pipeline directly

```bash
python scripts/run_pipeline.py path/to/clinical_assessment.wav
```

Prints the `FirstAssessment` JSON to stdout, and to stderr: the transcript,
segment count and duration, a per-field evidence audit, and a warning for any
field it couldn't ground.

## Tests

```bash
pytest -v            # 51 backend tests
cd frontend && npm test   # 37 frontend tests
```

All LLM and Mongo calls are mocked/faked — no `GROQ_API_KEY` or real MongoDB
connection is required to run either suite.

Backend coverage: the `FirstAssessment` schema, transcription (chunking,
timestamp shifting, caching, the retryable-vs-fatal error split), the
extraction graph (validation rules, the refine loop, sanitisation of bad model
output, grounding checks), the REST endpoints, and the Mongo repository.

Frontend coverage: `ConfidenceBadge`, `AssessmentView` read-only vs. editable
and the "+ Add manually" flow, `EvidencePanel` seeking and citation display,
the command palette (keyboard nav, filtering, registration lifecycle),
`api.ts`'s Zod validation and query-param construction, and an `UploadPage`
integration test covering parse → verify → edit → save plus flagged-section
and ungrounded-field handling.

## Known limitations

- Confidence flagging relies on the model self-reporting uncertain sections;
  it isn't a numeric confidence score from the API. Grounding checks catch the
  complementary failure — values the model was confident about but can't
  justify.
- Evidence is verified structurally (does the cited segment exist? is every
  populated field cited?), not semantically — a citation pointing at a real
  but irrelevant segment would pass. Showing the reviewer the cited text is
  the mitigation.
- Chunking assumes WAV; the API rejects other formats at the boundary.
- No auth on the endpoints — out of scope for this assignment.
