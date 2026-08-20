"""Preflight check: is this machine ready to run the service?

Everything that can go wrong on a fresh clone goes wrong *late* by default. A
missing API key surfaces after Whisper has spent three minutes transcribing; a
paused Atlas cluster surfaces when the clinician clicks Save, having already
reviewed the draft. Both are the same failure, a precondition that nobody
checked, and both are cheap to catch up front.

So this runs the checks in ascending order of cost and stops at the first one
that blocks, printing what to do about it rather than a traceback.

    python -m app.doctor           # cheap checks: seconds
    python -m app.doctor --warm    # also downloads Whisper weights, tests the key

The distinction matters on a first run: ``--warm`` pulls ~1.5 GB and spends one
Gemini request, which is a real cost on a free tier that allows five per minute.
Worth paying once, deliberately, rather than discovering both mid-demo.

Lives in ``app/`` rather than in the shell scripts because the checks are the
same on Windows and macOS, and logic written twice drifts.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Values shipped in .env.example that mean "you have not filled this in yet".
# Checking for emptiness alone is not enough: the example file has a working
# localhost Mongo URI, which is a real answer for some people and a forgotten
# placeholder for others. An empty GOOGLE_API_KEY is never intentional.
PLACEHOLDERS = {"", "your-key-here", "changeme", "<your-key>"}

OK, WARN, BAD = "  OK  ", " WARN ", " FAIL "


class Blocked(Exception):
    """A precondition that makes running the service pointless."""


def say(status: str, title: str, detail: str = "") -> None:
    print(f"[{status}] {title}")
    if detail:
        for line in detail.strip().splitlines():
            print(f"         {line}")


# --------------------------------------------------------------------------- #
# Checks, cheapest first
# --------------------------------------------------------------------------- #
def check_python() -> None:
    """3.14 is what the pins target, not a preference.

    Older pins for pymongo and faster-whisper have no cp314 wheels and try to
    build from source; the ones in requirements.txt are the versions that do.
    Running on 3.12 mostly works, so this warns rather than blocks.
    """
    major, minor = sys.version_info[:2]
    version = f"{major}.{minor}.{sys.version_info[2]}"
    if (major, minor) >= (3, 14):
        say(OK, f"Python {version}")
    elif (major, minor) >= (3, 11):
        say(WARN, f"Python {version}", "Built and tested on 3.14. Should work; untested.")
    else:
        raise Blocked(
            f"Python {version} is too old. This needs 3.11+, ideally 3.14.\n"
            "Install it from https://python.org and recreate the venv."
        )


def check_dependencies() -> None:
    """Import the heavy ones by name so a half-finished pip install is obvious."""
    required = {
        "fastapi": "fastapi",
        "pydantic_settings": "pydantic-settings",
        "pymongo": "pymongo",
        "faster_whisper": "faster-whisper",
        "langgraph": "langgraph",
        "langchain_google_genai": "langchain-google-genai",
    }
    missing = []
    for module, package in required.items():
        try:
            __import__(module)
        except ImportError:
            missing.append(package)

    if missing:
        raise Blocked(
            "Missing packages: " + ", ".join(missing) + "\n"
            "Run:  pip install -r requirements.txt"
        )
    say(OK, "Dependencies installed")


def check_env_file() -> None:
    """No .env means no key and no database, so this is a hard stop.

    Copying the example is something we could do silently, but then the run
    proceeds with an empty key and fails somewhere less obvious. Better to stop
    once, here, with the exact command.
    """
    env = ROOT / ".env"
    if env.exists():
        say(OK, ".env found")
        return

    example = ROOT / ".env.example"
    if example.exists():
        env.write_text(example.read_text(encoding="utf-8"), encoding="utf-8")
        raise Blocked(
            "No .env existed, so .env.example was copied to .env.\n"
            "Open it and fill in two values before running again:\n"
            "  GOOGLE_API_KEY:  free key from https://aistudio.google.com/apikey\n"
            "  MONGODB_URI:     Atlas connection string, or a local mongod"
        )
    raise Blocked("Neither .env nor .env.example exists. Is this the repository root?")


def check_settings():
    """Load config and confirm the two values a human has to supply are real."""
    from app.config import get_settings

    settings = get_settings()

    key = (settings.google_api_key or "").strip()
    if key in PLACEHOLDERS:
        raise Blocked(
            "GOOGLE_API_KEY is not set in .env\n"
            "Get a free key at https://aistudio.google.com/apikey"
        )
    # Masked: this prints to a terminal that may be shared or recorded.
    say(OK, f"GOOGLE_API_KEY set ({key[:4]}...{key[-4:]})")

    uri = (settings.mongodb_uri or "").strip()
    if uri in PLACEHOLDERS:
        raise Blocked("MONGODB_URI is not set in .env")
    say(OK, f"MONGODB_URI set ({_mask_uri(uri)})")

    say(OK, f"Whisper model: {settings.whisper_model} on {settings.whisper_device}")
    say(OK, f"Extraction model: {settings.extraction_model}")
    return settings


def _mask_uri(uri: str) -> str:
    """Never print Atlas credentials, even locally."""
    if "@" not in uri:
        return uri
    scheme, _, rest = uri.partition("://")
    return f"{scheme}://<credentials>@{rest.partition('@')[2]}"


def check_mongo() -> None:
    """An unreachable database is the single most common setup failure.

    Usually one of three things: the Atlas cluster is paused, the current IP is
    not on the access list, or a password with an @ in it was not URL-encoded.
    """
    import asyncio

    from app import db

    async def ping() -> bool:
        try:
            return await db.ping()
        finally:
            await db.close()

    if asyncio.run(ping()):
        say(OK, "MongoDB reachable")
        return

    raise Blocked(
        "Could not reach MongoDB. Usually one of:\n"
        "  - the Atlas cluster is paused (resume it in the Atlas UI)\n"
        "  - your IP is not in Atlas > Network Access\n"
        "  - a password containing @ : / or ? was not URL-encoded\n"
        "  - no local mongod is running, if the URI points at localhost"
    )


# --------------------------------------------------------------------------- #
# Expensive checks, only with --warm
# --------------------------------------------------------------------------- #
def warm_whisper(settings) -> None:
    """Download the weights now rather than during the first upload.

    ~1.5 GB into ~/.cache/huggingface/hub. Doing it here means the first real
    transcription is only slow because of CPU, not because of the network.
    """
    say(OK, f"Downloading Whisper '{settings.whisper_model}' weights (~1.5 GB, once)...")
    from app.transcription import _load_model

    _load_model()
    say(OK, "Whisper weights ready")


def warm_gemini(settings) -> None:
    """One tiny generation, to prove the key works before it matters.

    Spends a request from a five-per-minute free tier, which is why this is not
    in the default run. A wrong key discovered here costs seconds; discovered
    after transcription it costs minutes.
    """
    from langchain_google_genai import ChatGoogleGenerativeAI

    try:
        model = ChatGoogleGenerativeAI(
            model=settings.extraction_model,
            google_api_key=settings.google_api_key,
            temperature=0.0,
        )
        model.invoke("Reply with the single word: ready")
    except Exception as exc:  # noqa: BLE001 (any failure gets the same advice)
        raise Blocked(
            f"The Gemini key or model was rejected: {exc}\n"
            "Check GOOGLE_API_KEY, and that EXTRACTION_MODEL names a model your\n"
            "key can reach (https://aistudio.google.com/apikey)."
        ) from exc
    say(OK, f"Gemini reachable ({settings.extraction_model})")


# --------------------------------------------------------------------------- #
def main() -> int:
    parser = argparse.ArgumentParser(
        prog="python -m app.doctor",
        description="Check this machine can run the assessment service.",
    )
    parser.add_argument(
        "--warm",
        action="store_true",
        help="also download Whisper weights and test the Gemini key (slow, once)",
    )
    args = parser.parse_args()

    # The modules we call log their own failures with tracebacks. That is correct
    # in a server and wrong here, where a wall of pymongo internals buries the
    # one line that says what to do. This screen is the error message.
    logging.disable(logging.WARNING)

    print("\nChecking setup\n" + "-" * 58)
    try:
        check_python()
        check_dependencies()
        check_env_file()
        settings = check_settings()
        check_mongo()
        if args.warm:
            warm_gemini(settings)
            warm_whisper(settings)
    except Blocked as exc:
        print()
        say(BAD, "Setup incomplete", str(exc))
        print("-" * 58)
        return 1

    print("-" * 58)
    print("Ready. Start the service with:\n")
    print("    uvicorn app.main:app --reload\n")
    print("Then open http://localhost:8000/ui/\n")
    if not args.warm:
        print("Tip: 'python -m app.doctor --warm' pre-downloads the Whisper")
        print("     weights so the first upload is not also a 1.5 GB download.\n")
    return 0


if __name__ == "__main__":
    # os is imported for callers that set env vars before us; keep the reference
    # explicit so linters do not strip it.
    os.environ.setdefault("PYTHONUTF8", "1")
    raise SystemExit(main())
