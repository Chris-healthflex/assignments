# Voice/Note → Structured Clinical Assessment Form Filler

Turns a clinician-patient audio session into a structured `FirstAssessment`
JSON document, matching the exact format Stance Health's clinician frontend
consumes.

## Stack

- **FastAPI** — HTTP API
- **Groq Whisper API** (`whisper-large-v3`) — audio transcription
- **Groq `openai/gpt-oss-120b` via LangGraph** — two-node extraction pipeline
  (extract → check confidence)
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
  would. The swap was made because a Groq key was
  available and OpenAI's wasn't; both `transcription.py` and
  `extraction_graph.py` isolate the provider behind a small interface
  (`Groq` client / `StructuredLLM` protocol), so switching back to OpenAI is
  a localized change, not a rewrite.
- **A 2-node LangGraph graph over a single LLM call**: the `extract` node
  produces both the `FirstAssessment` and a list of sections it wasn't
  confident about; `check_confidence` is a plain conditional node that turns
  "too many uncertain sections" into a decision the API can act on. Keeping
  confidence-checking as its own node (rather than folding it into the
  prompt) makes the threshold testable without touching the LLM.
- **Never hallucinate**: the extraction prompt instructs the model to leave a
  section's fields empty and name the section in `low_confidence_sections`
  rather than invent a value. The API turns 2+ flagged sections into a 422
  instead of silently returning guessed data.
- **MongoDB Atlas over local Mongo**: no local `mongod`/Docker install needed.
- **`mongomock-motor` for tests**: full async Motor test coverage without
  touching a real database.

## Project layout

```
app/
  main.py                    FastAPI app, lifespan-managed Mongo client
  config.py                  env-based settings
  schemas/first_assessment.py  the FirstAssessment Pydantic models
  services/transcription.py    Groq Whisper API wrapper
  services/extraction_graph.py LangGraph extraction pipeline (Groq Llama)
  db/mongo.py                  Motor-backed repository
  api/assessments.py            the 4 REST endpoints
scripts/run_pipeline.py       CLI: WAV in, FirstAssessment JSON out
tests/                        pytest suite (schema, transcription, graph, API, Mongo)
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
  Add `?include_debug=true` (used only by the demo frontend below) to get
  `{assessment, transcript, confidence, is_low_confidence,
  low_confidence_sections}` instead, and to skip the 422 so a human can
  review and complete flagged sections rather than being blocked outright.
  The graded/default response shape is unchanged either way.
- `POST /assessments` — save a `FirstAssessment` JSON body → `{"id": ...}`
- `GET /assessments/{id}` — fetch a saved assessment
- `GET /assessments?date_from=...&date_to=...` — list, optionally filtered by
  `createdAt`

Interactive docs at `http://localhost:8000/docs`.

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

Two views: **New Assessment** (drop/pick a WAV → see the transcribed +
extracted `FirstAssessment`, sectioned and readable → save it) and
**Saved Assessments** (browse everything previously saved via `GET
/assessments`).

### Frontend architecture

```
frontend/src/
  main.tsx                  React Router setup, ErrorBoundary, ToastProvider
  App.tsx                   layout shell (header/nav) + <Outlet/>
  pages/
    UploadPage.tsx           "/"           — upload, review, save
    HistoryPage.tsx          "/history"    — saved assessments list
    AssessmentDetailPage.tsx "/history/:id"— saved assessment detail
  hooks/
    useParseAssessment.ts    parse mutation: status/result/error
    useSaveAssessment.ts     save mutation: status/savedId, double-submit-safe
    useAssessments.ts        list query: data/loading/error/refetch
    useAssessment.ts         by-id query: data/loading/error
  schemas.ts                 Zod schemas mirroring the FirstAssessment
                              Pydantic models — single source of truth for
                              src/types.ts (derived via z.infer)
  api.ts                     fetch wrappers; every response is validated
                              against schemas.ts before the caller sees it
  components/
    AssessmentView.tsx        read-only or editable render of a FirstAssessment
    ConfidenceBadge.tsx        coverage score + formula explainer
    ErrorBoundary.tsx          catches render errors, offers a way back
    ui/                        Button, Card, Badge, Toast — small design
                                system used everywhere instead of ad hoc
                                Tailwind strings
```

Real client-side routes (not tab state) — `/history/:id` is a shareable,
refresh-safe URL; browser back/forward works. API responses are validated at
the boundary with Zod (`api.ts`) rather than trusted via a type assertion, so
a backend contract change surfaces as a clear `ApiShapeError`, not a silent
bad render. Data-fetching lives in hooks, not components, so pages stay
mostly presentational.

### Frontend tests

```bash
cd frontend
npm test
```

Vitest + React Testing Library, mocking the `api` module — no live backend
needed. Covers: `ConfidenceBadge` rendering/formula toggle, `AssessmentView`
read-only vs. editable rendering and the "+ Add manually" flow, `api.ts`'s
Zod validation (including rejecting a malformed response), and an
`UploadPage` integration test covering the full parse → review → save path
and the flagged-section highlighting.

## Running the pipeline directly

```bash
python scripts/run_pipeline.py path/to/clinical_assessment.wav
```

Prints the transcript to stderr and the `FirstAssessment` JSON to stdout.

## Tests

```bash
pytest -v
```

All LLM and Mongo calls are mocked/faked in tests — no `GROQ_API_KEY` or
real MongoDB connection is required to run the suite.

## Known limitations

- Confidence flagging relies on the model self-reporting uncertain sections;
  it isn't a numeric confidence score from the API.
- No auth on the endpoints — out of scope for this assignment.
