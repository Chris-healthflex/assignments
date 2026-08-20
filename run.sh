#!/usr/bin/env bash
# ============================================================================
#  One-command setup and launch (macOS / Linux).
#
#    ./run.sh          create the venv, install, check the setup, serve
#    ./run.sh warm     the same, but also pre-download the Whisper weights
#    ./run.sh test     run the test suite
#    ./run.sh check    only run the preflight checks, do not serve
#
#  The service listens on http://localhost:8000. Once it answers, the review
#  interface, both API documentation views and two live endpoints open in the
#  browser by themselves. Set PORT first if 8000 is already taken, or
#  OPEN_BROWSER=0 to start the server without opening anything:
#
#    PORT=8080 ./run.sh
#    OPEN_BROWSER=0 ./run.sh
#
#  Safe to run repeatedly: the venv and the installed packages are reused, so
#  a second run reaches the server in seconds.
#
#  Windows: use run.bat, which does the same thing.
# ============================================================================
set -euo pipefail

# Work from the repository root regardless of where this was invoked.
cd "$(dirname "${BASH_SOURCE[0]}")"

VENV=".venv"
PY="$VENV/bin/python"
STAMP="$VENV/.deps-installed"

# Set here rather than left to uvicorn's default, so the address printed below
# is the address actually being served. Respects an existing PORT, which is the
# way out when 8000 is occupied.
# Exported, not just assigned: the preflight below is a separate process and
# prints this address too.
export PORT="${PORT:-8000}"
URL="http://localhost:$PORT"

echo
echo " Clinical First Assessment - setup and launch"
echo " ============================================"
echo

# ---------------------------------------------------------------------------
# 1. Find a Python interpreter.
# ---------------------------------------------------------------------------
if [ ! -x "$PY" ]; then
    BOOTSTRAP=""
    for candidate in python3.14 python3.13 python3 python; do
        if command -v "$candidate" >/dev/null 2>&1; then
            BOOTSTRAP="$candidate"
            break
        fi
    done

    if [ -z "$BOOTSTRAP" ]; then
        echo " [FAIL] No Python interpreter found."
        echo
        echo "        macOS:  brew install python@3.14"
        echo "        Linux:  sudo apt install python3.14 python3.14-venv"
        echo
        exit 1
    fi

    echo " [ .. ] Creating virtual environment in $VENV"
    "$BOOTSTRAP" -m venv "$VENV"
    echo " [ OK ] Virtual environment created"
fi

# ---------------------------------------------------------------------------
# 2. Install dependencies.
#    The stamp is a *copy* of requirements.txt, compared byte for byte rather
#    than by timestamp, because git checkout rewrites mtimes and would make an
#    unchanged file look newer than the stamp.
# ---------------------------------------------------------------------------
if ! cmp -s requirements.txt "$STAMP" 2>/dev/null; then
    echo " [ .. ] Installing dependencies (a few minutes on a first run)"
    "$PY" -m pip install --upgrade pip --quiet
    if ! "$PY" -m pip install -r requirements.txt; then
        echo
        echo " [FAIL] Dependency installation failed. Scroll up for the reason."
        echo "        A common cause is an older Python without prebuilt wheels"
        echo "        for these pins - 3.14 is what they target."
        exit 1
    fi
    cp requirements.txt "$STAMP"
    echo " [ OK ] Dependencies installed"
else
    echo " [ OK ] Dependencies already installed"
fi

# ---------------------------------------------------------------------------
# 3. Dispatch on the requested mode.
# ---------------------------------------------------------------------------
serve() {
    echo
    echo " ============================================================"
    echo "  Running at  $URL/ui/"
    echo " ============================================================"
    echo
    echo "  Review interface   $URL/ui/"
    echo "  API explorer       $URL/docs"
    echo "  API reference      $URL/redoc"
    echo "  Health check       $URL/health"
    echo "  Saved assessments  $URL/assessments"
    echo
    echo "  Press Ctrl+C to stop."
    echo
    # Opens the tabs above once the server answers. Backgrounded so it waits
    # alongside uvicorn rather than before it: the port is not bound yet at this
    # point, so opening them here would land on a dead socket.
    if [ "${OPEN_BROWSER:-1}" != "0" ]; then
        "$PY" -m app.browser &
    fi
    # Bound to 127.0.0.1 rather than 0.0.0.0 on purpose: this process handles
    # patient audio, and it has no business being reachable from the network
    # until someone decides that deliberately.
    exec "$PY" -m uvicorn app.main:app --reload --host 127.0.0.1 --port "$PORT"
}

case "${1:-serve}" in
    test)
        exec "$PY" -m pytest
        ;;
    check)
        exec "$PY" -m app.doctor
        ;;
    warm)
        # The doctor prints what is missing and why; a second message here
        # would only bury it, so just stop on a non-zero exit.
        "$PY" -m app.doctor --warm
        serve
        ;;
    *)
        "$PY" -m app.doctor
        serve
        ;;
esac
