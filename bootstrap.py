"""One-command setup for the clinical assessment pipeline.

Creates the virtual environment, installs dependencies, starts MongoDB if Docker
is available, seeds the env file, and proves the install by running the suite::

    python bootstrap.py

Safe to re-run: an existing environment or container is reused, and an existing
``.env`` is never overwritten.
"""

import os
import shutil
import subprocess
import sys
import venv
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
VENV_DIR = REPO_ROOT / ".venv"
BACKEND_DIR = REPO_ROOT / "backend"
MONGO_CONTAINER = "stance-mongo"
MONGO_IMAGE = "mongo:7"
PLACEHOLDER_KEY = "replace-with-your-key"
MINIMUM_PYTHON = (3, 11)
TOTAL_STEPS = 5


def venv_python_path(venv_dir: Path) -> Path:
    """Return the interpreter inside a virtual environment.

    The only platform-dependent path in this script, isolated here so the rest
    stays platform-agnostic.

    Args:
        venv_dir: Root of the virtual environment.

    Returns:
        Path to the interpreter, which may not exist yet.
    """
    if sys.platform == "win32":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def announce(step: int, message: str) -> None:
    """Print a numbered progress line."""
    print(f"\n[{step}/{TOTAL_STEPS}] {message}", flush=True)


def run(command: list[str], cwd: Path | None = None) -> None:
    """Run a command, failing loudly with its exit status.

    Args:
        command: Argument vector to execute.
        cwd: Working directory, or None for the current one.

    Raises:
        SystemExit: If the command exits non-zero.
    """
    result = subprocess.run(command, cwd=cwd, check=False)
    if result.returncode != 0:
        raise SystemExit(f"failed ({result.returncode}): {' '.join(command)}")


def ensure_supported_python() -> None:
    """Stop early if the interpreter is too old.

    Raises:
        SystemExit: If the running interpreter predates the minimum.
    """
    if sys.version_info < MINIMUM_PYTHON:
        required = ".".join(str(part) for part in MINIMUM_PYTHON)
        raise SystemExit(
            f"Python {required}+ is required, found {sys.version.split()[0]}"
        )


def ensure_virtualenv() -> Path:
    """Create the virtual environment if it is missing, and return its Python."""
    interpreter = venv_python_path(VENV_DIR)
    if interpreter.exists():
        print(f"      reusing {VENV_DIR.name}")
        return interpreter

    # with_pip because the very next step shells out to pip.
    venv.create(VENV_DIR, with_pip=True)
    print(f"      created {VENV_DIR.name}")
    return interpreter


def install_dependencies(interpreter: Path) -> None:
    """Install pinned dependencies and the package itself, editable."""
    run([str(interpreter), "-m", "pip", "install", "--quiet", "--upgrade", "pip"])
    run(
        [
            str(interpreter),
            "-m",
            "pip",
            "install",
            "--quiet",
            "-r",
            str(BACKEND_DIR / "requirements.txt"),
        ]
    )
    run([str(interpreter), "-m", "pip", "install", "--quiet", "-e", str(BACKEND_DIR)])
    print("      dependencies installed")


def docker_is_running() -> bool:
    """Report whether a Docker daemon is reachable."""
    if shutil.which("docker") is None:
        return False
    probe = subprocess.run(
        ["docker", "info"], capture_output=True, text=True, check=False
    )
    return probe.returncode == 0


def start_mongodb() -> bool:
    """Start or reuse the MongoDB container.

    Returns:
        True if MongoDB is available afterwards, False if Docker is not.
    """
    if not docker_is_running():
        print("      Docker not available - skipping MongoDB")
        print("      Storage endpoints (EP2/EP3/EP4) will return 503 until it is")
        return False

    existing = subprocess.run(
        ["docker", "ps", "-aq", "-f", f"name=^{MONGO_CONTAINER}$"],
        capture_output=True,
        text=True,
        check=False,
    )
    if existing.stdout.strip():
        # capture_output because `docker start` echoes the container name.
        subprocess.run(
            ["docker", "start", MONGO_CONTAINER], capture_output=True, check=True
        )
        print(f"      reusing container {MONGO_CONTAINER}")
        return True

    run(
        [
            "docker",
            "run",
            "-d",
            "-p",
            "27017:27017",
            "--name",
            MONGO_CONTAINER,
            MONGO_IMAGE,
        ]
    )
    print(f"      started {MONGO_IMAGE} as {MONGO_CONTAINER}")
    return True


def seed_env_file() -> None:
    """Create ``backend/.env`` from the example if it does not exist.

    An existing file is left alone: it may already hold a real API key.
    """
    env_file = BACKEND_DIR / ".env"
    if env_file.exists():
        print("      backend/.env already exists - left untouched")
        return
    shutil.copyfile(BACKEND_DIR / ".env.example", env_file)
    print("      created backend/.env from the example")


def _key_in_file(env_file: Path) -> bool:
    """Report whether an env file sets a real-looking API key."""
    if not env_file.is_file():
        return False
    for line in env_file.read_text(encoding="utf-8").splitlines():
        if line.startswith("ANTHROPIC_API_KEY="):
            value = line.partition("=")[2].strip()
            return bool(value) and PLACEHOLDER_KEY not in value
    return False


def api_key_is_configured() -> bool:
    """Report whether a key is set anywhere config.py would find it.

    Mirrors config.py's precedence - exported variable, then ``.env.local``,
    then ``.env`` - so setup does not nag someone who has already set it.
    """
    if os.environ.get("ANTHROPIC_API_KEY", "").strip():
        return True
    return any(_key_in_file(BACKEND_DIR / name) for name in (".env.local", ".env"))


def run_test_suite(interpreter: Path) -> None:
    """Run the unit suite, so setup ends in proof rather than a claim."""
    run([str(interpreter), "-m", "pytest", "-q"], cwd=BACKEND_DIR)


def report_next_steps(*, has_key: bool, has_mongo: bool) -> None:
    """Print what the operator still has to do, if anything."""
    print("\nSetup complete.\n")
    if not has_key:
        print("  ! Set your key in backend/.env before running the pipeline:")
        print("      ANTHROPIC_API_KEY=sk-ant-...")
    if not has_mongo:
        print(
            "  ! Start Docker, then: docker run -d -p 27017:27017 "
            f"--name {MONGO_CONTAINER} {MONGO_IMAGE}"
        )

    interpreter = venv_python_path(VENV_DIR)
    print("\n  Run the pipeline:")
    print(
        f"      {interpreter} backend/scripts/run_pipeline.py "
        "clinical_assessment.wav > assessment.json"
    )
    print("\n  Run the API:")
    print(f"      {interpreter} -m uvicorn clinical_assessment.api:app --reload")


def main() -> int:
    """Set the project up end to end.

    Returns:
        0 on success; non-zero exits are raised as SystemExit by :func:`run`.
    """
    print("Clinical Assessment Pipeline - setup")
    ensure_supported_python()

    announce(1, "Virtual environment")
    interpreter = ensure_virtualenv()

    announce(2, "Dependencies")
    install_dependencies(interpreter)

    announce(3, "MongoDB")
    has_mongo = start_mongodb()

    announce(4, "Environment file")
    seed_env_file()
    has_key = api_key_is_configured()

    announce(5, "Verifying with the test suite")
    run_test_suite(interpreter)

    report_next_steps(has_key=has_key, has_mongo=has_mongo)
    return 0


if __name__ == "__main__":
    sys.exit(main())
