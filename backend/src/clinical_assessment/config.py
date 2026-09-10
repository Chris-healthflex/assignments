"""Settings read from the environment, with defaults for local development.

Deliberately a plain frozen dataclass over ``os.environ`` rather than a settings
framework: the pipeline has four knobs, and every one of them has a working
default.
"""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# `medium` over `small`: on the supplied recording `small` heard "knee gig 5"
# where the clinician said "negative five degrees", destroying a measurement
# rather than merely garbling a word. It is ~10x slower on CPU, which a batch
# pipeline over a two-minute recording can afford.
DEFAULT_WHISPER_MODEL_SIZE = "medium"
DEFAULT_CONFIDENCE_THRESHOLD = 0.70
DEFAULT_MONGODB_URI = "mongodb://localhost:27017"
DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-5"

# src/clinical_assessment/config.py -> the backend directory holding the env files.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE_NAMES = (".env.local", ".env")


def load_env_files(directory: Path) -> None:
    """Load ``.env.local`` then ``.env`` from a directory, if they exist.

    Precedence is exported variables, then ``.env.local``, then ``.env``. That
    falls out of loading the more specific file first with ``override=False``:
    the first value seen for a name wins, and anything already exported was set
    before either file was read.

    Args:
        directory: Directory to look for the env files in. Missing files are
            ignored, so a deployment that exports real environment variables
            needs no files at all.
    """
    for file_name in ENV_FILE_NAMES:
        env_path = directory / file_name
        if env_path.is_file():
            load_dotenv(env_path, override=False)


# Read at import so that every entry point - the script, uvicorn, the tests -
# sees the same values without each having to remember to call this.
load_env_files(PROJECT_ROOT)


@dataclass(frozen=True)
class Settings:
    """Runtime configuration for the assessment pipeline.

    Attributes:
        whisper_model_size: Whisper checkpoint name, e.g. ``"small"``.
        confidence_threshold: Grounded-field ratio at or above which an
            extraction is accepted.
        mongodb_uri: Connection string for the assessment store.
        anthropic_model: Model identifier used for extraction.
    """

    whisper_model_size: str
    confidence_threshold: float
    mongodb_uri: str
    anthropic_model: str


def load_settings() -> Settings:
    """Read settings from the environment, falling back to module defaults.

    Returns:
        The resolved settings.

    Raises:
        ValueError: If ``CONFIDENCE_THRESHOLD`` is set to a non-numeric value or
            falls outside the ratio range [0.0, 1.0].
    """
    return Settings(
        whisper_model_size=os.environ.get(
            "WHISPER_MODEL_SIZE", DEFAULT_WHISPER_MODEL_SIZE
        ),
        confidence_threshold=_read_confidence_threshold(),
        mongodb_uri=os.environ.get("MONGODB_URI", DEFAULT_MONGODB_URI),
        anthropic_model=os.environ.get("ANTHROPIC_MODEL", DEFAULT_ANTHROPIC_MODEL),
    )


def _read_confidence_threshold() -> float:
    """Parse the confidence threshold, validating it as a ratio."""
    raw_threshold = os.environ.get(
        "CONFIDENCE_THRESHOLD", str(DEFAULT_CONFIDENCE_THRESHOLD)
    )
    try:
        threshold = float(raw_threshold)
    except ValueError as exc:
        raise ValueError(
            f"CONFIDENCE_THRESHOLD must be a number, got {raw_threshold!r}"
        ) from exc

    # A threshold outside [0, 1] can never be compared meaningfully against a
    # grounded-field ratio, so it is a configuration error rather than a
    # permanently open or permanently closed gate.
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(
            f"CONFIDENCE_THRESHOLD must be between 0.0 and 1.0, got {threshold}"
        )
    return threshold
