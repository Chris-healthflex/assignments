"""End-to-end run: audio -> transcript -> assessment -> MongoDB -> back again.

    python -m tests.run_pipeline clinical_assessment.wav
    python -m tests.run_pipeline clinical_assessment.wav --http
    python -m tests.run_pipeline --transcript transcript.txt --no-save
    python -m tests.run_pipeline clinical_assessment.wav > assessment.json

Narration goes to stderr and the `FirstAssessment` JSON goes to stdout, so the
last form above writes a clean contract document to a file while still showing
progress in the terminal.

Two ways to drive the same pipeline. By default the stages are called in
process, which is the shortest path to a JSON document and needs no server. With
`--http` the identical work happens through the running API, which is the one
that proves the endpoints, the 422 and the persistence layer agree with each
other rather than merely working alone.

Exit codes: 0 if every stage passed, 1 if one failed. Low confidence is not a
failure -- a 422 is the service working correctly -- but an extraction that
produced nothing at all is, because that is indistinguishable from success.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any

from app.schemas import SECTIONS, ExtractionFlags, FirstAssessment, StoredAssessment

# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #
BAR = "-" * 72


def say(message: str = "") -> None:
    """Narration. stderr, so stdout stays a clean JSON document."""
    print(message, file=sys.stderr, flush=True)


def stage(number: int, title: str) -> None:
    say(f"\n{BAR}\n{number}. {title}\n{BAR}")


def ok(message: str) -> None:
    say(f"  ok    {message}")


def bad(message: str) -> None:
    say(f"  FAIL  {message}")


# --------------------------------------------------------------------------- #
# The contract check
# --------------------------------------------------------------------------- #
def _nulls(node: Any, path: str = "") -> list[str]:
    """Every place a null slipped into what should be a string."""
    if node is None:
        return [path or "(root)"]
    if isinstance(node, dict):
        return [p for key, value in node.items() for p in _nulls(value, f"{path}.{key}" if path else key)]
    if isinstance(node, list):
        return [p for i, item in enumerate(node) for p in _nulls(item, f"{path}[{i}]")]
    return []


def check_contract(payload: dict[str, Any]) -> list[str]:
    """The brief's three hard rules, checked against the JSON actually produced.

    Not against the model -- against the serialised document, because that is
    what the frontend receives and the only thing the "exact match" promise is
    really about.
    """
    problems: list[str] = []

    keys = tuple(payload.keys())
    if keys != SECTIONS:
        missing = [s for s in SECTIONS if s not in keys]
        extra = [k for k in keys if k not in SECTIONS]
        if missing:
            problems.append(f"missing section(s): {', '.join(missing)}")
        if extra:
            problems.append(f"unexpected key(s): {', '.join(extra)}")
        if not missing and not extra:
            problems.append(f"sections out of order: {keys}")

    nulls = _nulls(payload)
    if nulls:
        problems.append(f"null where a string was required: {', '.join(nulls[:5])}")

    # Arrays stay arrays, even holding a single item.
    for key in ("subjectiveAssessments", "subjectiveGoals", "objectiveGoals", "recommendation"):
        if key in payload and not isinstance(payload[key], list):
            problems.append(f"{key} is {type(payload[key]).__name__}, not a list")
    tests = payload.get("objectiveAssessment", {}).get("tests")
    if tests is not None and not isinstance(tests, list):
        problems.append("objectiveAssessment.tests is not a list")

    # Re-validating the document must produce the same document. If it does not,
    # something is being coerced on the way through and the JSON on the wire is
    # not what the model believes it emitted.
    try:
        if FirstAssessment.model_validate(payload).model_dump(mode="json") != payload:
            problems.append("the document does not survive a re-validation unchanged")
    except Exception as exc:  # noqa: BLE001 -- any failure here is a contract failure
        problems.append(f"the document does not validate against FirstAssessment: {exc}")

    return problems


def report_confidence(flags: ExtractionFlags, threshold: float) -> list[str]:
    """Print the per-field picture and return the paths that fall short."""
    fields = flags.fields
    if not fields:
        say("  no evidence was recorded for any field")
        return []

    failing = flags.below(threshold)
    say(f"  {len(fields)} field(s) with evidence, overall {flags.overallConfidence:.0%}")
    say(f"  {len(flags.ungrounded())} without a traceable source, {len(failing)} below {threshold:.0%}")

    if failing:
        say("")
        for field in sorted(failing, key=lambda f: f.confidence):
            signals = " ".join(
                f"{name} {value:.2f}" if value is not None else f"{name} --"
                for name, value in (
                    ("model", field.modelConfidence),
                    ("audio", field.audioConfidence),
                    ("ctx", field.contextConfidence),
                )
            )
            say(f"  {field.confidence:>5.2f}  {field.field:<44} {field.value[:22]!r:<24} {signals}")
            if field.reason:
                say(f"         {field.reason}")
    return [f.field for f in failing]


# --------------------------------------------------------------------------- #
# In-process run
# --------------------------------------------------------------------------- #
async def run_direct(audio: Path | None, transcript_file: Path | None, save: bool) -> tuple[int, dict]:
    from app.config import get_settings
    from app.extraction import ExtractionFailed, extract
    from app.transcription import TranscriptionError, transcribe

    settings = get_settings()
    failures = 0

    stage(1, "Transcription")
    transcription = None
    if transcript_file is not None:
        transcript = transcript_file.read_text(encoding="utf-8").strip()
        ok(f"loaded {transcript_file} ({len(transcript)} characters)")
        say("  no word-level audio confidence without the recording; scores will be model-only")
    else:
        try:
            transcription = transcribe(audio)
        except (TranscriptionError, NotImplementedError) as exc:
            bad(str(exc))
            return 1, {}
        transcript = transcription.text
        words = sum(len(segment.words) for segment in transcription.segments)
        ok(f"{audio.name}: {transcription.durationSec:.1f}s, {words} words, language {transcription.language!r}")
    say(f"\n  {transcript[:220]}{'...' if len(transcript) > 220 else ''}")

    stage(2, "Extraction")
    try:
        result = extract(transcript, transcription)
    except ExtractionFailed as exc:
        bad(f"extraction failed: {exc}")
        return 1, {}
    ok(f"model {settings.extraction_model}")

    payload = result.assessment.model_dump(mode="json")

    # A run where every model call failed still yields a valid, empty document.
    # Without this check it would print as a clean success, which is the one
    # outcome this pipeline must never report.
    if not result.flags.fields and not any(_populated(payload).values()):
        bad("nothing was extracted at all -- an empty result is not a successful run")
        say("  check the log above for 429 / quota errors from the model provider")
        return 1, payload

    stage(3, "Contract")
    problems = check_contract(payload)
    for problem in problems:
        bad(problem)
    if not problems:
        ok(f"exactly the {len(SECTIONS)} required sections, in order, no nulls, arrays intact")
    failures += len(problems)

    say("")
    for section, count in _populated(payload).items():
        say(f"    {section:<24} {count}")

    stage(4, "Confidence")
    report_confidence(result.flags, settings.extraction_confidence_threshold)

    if not save:
        say("\n  --no-save: skipping MongoDB")
        return (1 if failures else 0), payload

    failures += await persist_and_verify(
        StoredAssessment(
            audioFilename=audio.name if audio else (transcript_file.name if transcript_file else ""),
            transcript=transcript,
            flags=result.flags,
            assessment=result.assessment,
        ),
        payload,
    )
    return (1 if failures else 0), payload


def _populated(payload: dict[str, Any]) -> dict[str, str]:
    """A one-line census of each section, for the eye rather than for a test."""
    census: dict[str, str] = {}
    for key, value in payload.items():
        if key == "objectiveAssessment":
            census[key] = f"{len(value.get('tests', []))} test(s)"
        elif isinstance(value, list):
            census[key] = f"{len(value)} item(s)"
        else:
            filled = sum(1 for v in value.values() if v)
            census[key] = f"{filled}/{len(value)} field(s) populated"
    return census


async def persist_and_verify(stored: StoredAssessment, payload: dict[str, Any]) -> int:
    """Save, read back, and confirm the contract survived the database."""
    from pymongo.errors import PyMongoError

    from app import db

    failures = 0
    stage(5, "MongoDB")
    try:
        if not await db.ping():
            bad("no MongoDB at MONGODB_URI -- rerun with --no-save to skip persistence")
            return 1

        await db.ensure_indexes()
        assessment_id = await db.save_assessment(stored)
        ok(f"saved as {assessment_id}")

        fetched = await db.get_assessment(assessment_id)
        if fetched is None:
            bad("saved, but could not be read back")
            return 1

        if fetched.assessment.model_dump(mode="json") != payload:
            bad("the contract changed on the way through the database")
            failures += 1
        else:
            ok("read back byte for byte")

        if abs(fetched.flags.overallConfidence - stored.flags.overallConfidence) > 1e-9:
            bad("confidence did not survive the round trip")
            failures += 1
        else:
            ok(f"confidence recomputed on read: {fetched.flags.overallConfidence:.0%}")

        day = fetched.createdAt.date()
        listed = await db.list_assessments(day)
        if any(row.id == assessment_id for row in listed):
            ok(f"appears in GET /assessments?date={day} ({len(listed)} that day)")
        else:
            bad(f"saved but absent from the {day} listing")
            failures += 1

        elsewhere = await db.list_assessments(date(2000, 1, 1))
        if elsewhere:
            bad(f"the date filter returned {len(elsewhere)} row(s) for 2000-01-01")
            failures += 1
        else:
            ok("the date filter excludes other days")
    except PyMongoError as exc:
        bad(f"database error: {exc}")
        failures += 1
    finally:
        await db.close()
    return failures


# --------------------------------------------------------------------------- #
# Over HTTP
# --------------------------------------------------------------------------- #
async def run_http(audio: Path, base_url: str, save: bool) -> tuple[int, dict]:
    import httpx

    from app.config import get_settings

    failures = 0
    threshold = get_settings().extraction_confidence_threshold

    async with httpx.AsyncClient(base_url=base_url, timeout=900.0) as client:
        stage(1, "Service")
        try:
            health = await client.get("/health")
        except httpx.HTTPError as exc:
            bad(f"no service at {base_url}: {exc}")
            say("  start one with: uvicorn app.main:app --reload")
            return 1, {}
        body = health.json()
        (ok if health.status_code == 200 else bad)(f"/health {health.status_code} {body}")
        failures += health.status_code != 200

        stage(2, "POST /assessments/parse")
        with audio.open("rb") as handle:
            response = await client.post(
                "/assessments/parse", files={"file": (audio.name, handle, "audio/wav")}
            )
        if response.status_code not in (200, 422):
            bad(f"unexpected {response.status_code}: {response.text[:300]}")
            return 1, {}

        parsed = response.json()
        detail = parsed.get("detail") if response.status_code == 422 else []
        ok(f"{response.status_code} with {len(detail or [])} field(s) held back for review")

        payload = parsed["assessment"]
        flags = ExtractionFlags.model_validate(parsed["flags"])

        if response.status_code == 422:
            say("")
            for error in detail:
                loc = ".".join(str(part) for part in error["loc"][1:])
                ctx = error.get("ctx", {})
                say(f"  {ctx.get('confidence', 0):>5.2f}  {error['type']:<20} {loc}")
                say(f"         {error['msg']}")
            # The 422 must point at fields that exist in the document it returned.
            for error in detail:
                if _resolve(payload, error["loc"][1:]) is _MISSING:
                    bad(f"detail points at a field that is not in the response: {error['loc']}")
                    failures += 1

        stage(3, "Contract")
        problems = check_contract(payload)
        for problem in problems:
            bad(problem)
        if not problems:
            ok("the JSON on the wire matches the contract exactly")
        failures += len(problems)

        stage(4, "Confidence")
        report_confidence(flags, threshold)

        if not save:
            say("\n  --no-save: skipping the write endpoints")
            return (1 if failures else 0), payload

        stage(5, "POST /assessments, then read it back")
        saved = await client.post(
            "/assessments",
            json={
                "audioFilename": parsed.get("audioFilename") or audio.name,
                "transcript": parsed.get("transcript", ""),
                "flags": parsed["flags"],
                "assessment": payload,
            },
        )
        if saved.status_code != 201:
            bad(f"save returned {saved.status_code}: {saved.text[:300]}")
            return 1, payload
        record = saved.json()
        ok(f"201, id {record['id']}")

        fetched = await client.get(f"/assessments/{record['id']}")
        if fetched.status_code != 200:
            bad(f"GET by id returned {fetched.status_code}")
            failures += 1
        elif fetched.json()["assessment"] != payload:
            bad("the contract changed between save and fetch")
            failures += 1
        else:
            ok("GET /assessments/{id} returns it byte for byte")

        day = record["createdAt"][:10]
        listed = await client.get("/assessments", params={"date": day})
        if any(row["id"] == record["id"] for row in listed.json()):
            ok(f"GET /assessments?date={day} includes it")
        else:
            bad(f"absent from the {day} listing")
            failures += 1

        empty = await client.get("/assessments", params={"date": "2000-01-01"})
        if empty.json():
            bad("the date filter returned rows for 2000-01-01")
            failures += 1
        else:
            ok("the date filter excludes other days")

        stage(6, "Rejections")
        for label, files, expected in (
            ("a PDF", {"file": ("notes.pdf", b"%PDF-1.4", "application/pdf")}, 415),
            ("an empty file", {"file": ("empty.wav", b"", "audio/wav")}, 400),
        ):
            got = (await client.post("/assessments/parse", files=files)).status_code
            (ok if got == expected else bad)(f"{label} -> {got} (expected {expected})")
            failures += got != expected

        for label, path, expected in (
            ("a malformed id", "/assessments/not-an-id", 404),
            ("an unknown id", "/assessments/507f1f77bcf86cd799439011", 404),
        ):
            got = (await client.get(path)).status_code
            (ok if got == expected else bad)(f"{label} -> {got} (expected {expected})")
            failures += got != expected

        got = (await client.get("/assessments", params={"date": "yesterday"})).status_code
        (ok if got == 422 else bad)(f"an unparseable date -> {got} (expected 422)")
        failures += got != 422

    return (1 if failures else 0), payload


_MISSING = object()


def _resolve(node: Any, steps: list[Any]) -> Any:
    for step in steps:
        try:
            node = node[step]
        except (KeyError, IndexError, TypeError):
            return _MISSING
    return node


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the whole pipeline and print the FirstAssessment JSON.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("audio", nargs="?", type=Path, help="Recording to process")
    parser.add_argument("--transcript", type=Path, help="Use an existing transcript instead of Whisper")
    parser.add_argument(
        "--http",
        nargs="?",
        const="http://127.0.0.1:8000",
        metavar="URL",
        help="Drive a running service over HTTP instead of calling the stages in process",
    )
    parser.add_argument("--no-save", action="store_true", help="Skip MongoDB")
    parser.add_argument("--out", type=Path, help="Also write the assessment JSON here")
    args = parser.parse_args()

    if args.http and args.transcript:
        parser.error("--http uploads a recording; it cannot start from a transcript")
    if args.http and args.audio is None:
        parser.error("--http needs an audio file to upload")
    if not args.http and args.audio is None and args.transcript is None:
        parser.error("provide an audio file, or --transcript FILE")
    if args.audio is not None and not args.audio.is_file():
        parser.error(f"no such file: {args.audio}")

    if args.http:
        code, payload = asyncio.run(run_http(args.audio, args.http.rstrip("/"), not args.no_save))
    else:
        code, payload = asyncio.run(run_direct(args.audio, args.transcript, not args.no_save))

    if payload:
        stage(9, "FirstAssessment")
        say("  (the document below is on stdout; everything else is on stderr)\n")
        document = json.dumps(payload, indent=2, ensure_ascii=False)
        print(document)
        if args.out:
            args.out.write_text(document + "\n", encoding="utf-8")
            say(f"\n  written to {args.out}")

    say(f"\n{BAR}")
    say("PIPELINE OK" if code == 0 else "PIPELINE FAILED")
    say(BAR)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
