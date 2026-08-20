"""Open the service in a browser once it is actually answering.

`run.bat` and `run.sh` start this in the background immediately before handing
the terminal to uvicorn. Opening the tabs straight away would race the server:
uvicorn needs a moment to bind, and with `--reload` it also has a supervisor to
spawn first, so the tabs would land on a connection error and the evaluator
would be looking at a browser failure page while a perfectly healthy service
started up behind it. So this waits for `/health` to answer before opening
anything, and gives up quietly rather than opening tabs onto a dead port.

Shared by both scripts for the same reason `app.doctor` is: one implementation
that behaves identically on Windows and macOS beats two that drift.

    OPEN_BROWSER=0   start the server without opening anything
    OPEN_TABS=...    comma-separated paths, if the default set is not wanted
"""

from __future__ import annotations

import os
import sys
import time
import urllib.error
import urllib.request
import webbrowser

# The review interface first, because that is what the service is for and it is
# the tab the browser lands on. Then the two documentation views FastAPI serves,
# then a live endpoint each for the service and the stored data, so every part
# of the brief is on screen without anyone having to type a URL.
DEFAULT_TABS = ("/ui/", "/docs", "/redoc", "/health", "/assessments")

# Whisper is not involved in booting, so this is only covering process start and
# index creation. Long enough for a cold Atlas connection, short enough that a
# server which is never coming up does not leave something waiting all day.
STARTUP_TIMEOUT_SEC = 90.0
POLL_SEC = 0.5

# Browsers drop tabs that arrive in a single burst, particularly when the first
# one is also launching the browser itself.
BETWEEN_TABS_SEC = 0.4


def _answering(base: str) -> bool:
    """Has the server started listening?

    Any HTTP response counts, including the 503 that `/health` returns when
    MongoDB is unreachable. That is the service correctly reporting a degraded
    dependency, and it is exactly the sort of thing worth opening the browser to
    show rather than hiding behind a silent failure to launch.
    """
    try:
        urllib.request.urlopen(f"{base}/health", timeout=2)
    except urllib.error.HTTPError:
        return True
    except OSError:
        return False
    return True


def tabs() -> tuple[str, ...]:
    raw = os.environ.get("OPEN_TABS", "")
    if not raw.strip():
        return DEFAULT_TABS
    return tuple(p if p.startswith("/") else f"/{p}" for p in raw.split(",") if p.strip())


def main() -> int:
    if os.environ.get("OPEN_BROWSER", "1") == "0":
        return 0

    base = f"http://localhost:{os.environ.get('PORT', '8000')}"
    deadline = time.monotonic() + STARTUP_TIMEOUT_SEC

    while time.monotonic() < deadline:
        if _answering(base):
            for index, path in enumerate(tabs()):
                webbrowser.open_new_tab(base + path)
                if index:
                    time.sleep(BETWEEN_TABS_SEC)
            return 0
        time.sleep(POLL_SEC)

    # Nothing is printed on the happy path: uvicorn owns the terminal by now and
    # this process has no business writing over its log.
    print(
        f"Could not reach {base} within {STARTUP_TIMEOUT_SEC:.0f}s, so no tabs were opened.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
