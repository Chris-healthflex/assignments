"""Tests for environment-driven settings."""

import os
from pathlib import Path

import pytest

from clinical_assessment.config import (
    DEFAULT_ANTHROPIC_MODEL,
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_MONGODB_URI,
    DEFAULT_WHISPER_MODEL_SIZE,
    Settings,
    load_env_files,
    load_settings,
)

SETTING_ENV_VARS = [
    "WHISPER_MODEL_SIZE",
    "CONFIDENCE_THRESHOLD",
    "MONGODB_URI",
    "ANTHROPIC_MODEL",
]


@pytest.fixture(autouse=True)
def clear_setting_env_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    """Isolate every test from whatever the developer has exported."""
    for name in SETTING_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def test_defaults_apply_when_nothing_is_exported() -> None:
    settings = load_settings()

    assert settings == Settings(
        whisper_model_size=DEFAULT_WHISPER_MODEL_SIZE,
        confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD,
        mongodb_uri=DEFAULT_MONGODB_URI,
        anthropic_model=DEFAULT_ANTHROPIC_MODEL,
    )


def test_environment_overrides_every_setting(monkeypatch: pytest.MonkeyPatch) -> None:
    # Every value here is deliberately not the default, so the assertion proves
    # the override works rather than passing on a value that would have been
    # returned anyway.
    monkeypatch.setenv("WHISPER_MODEL_SIZE", "large")
    monkeypatch.setenv("CONFIDENCE_THRESHOLD", "0.85")
    monkeypatch.setenv("MONGODB_URI", "mongodb://example:27017")
    monkeypatch.setenv("ANTHROPIC_MODEL", "claude-opus-5")

    settings = load_settings()

    assert (
        settings.whisper_model_size,
        settings.confidence_threshold,
        settings.mongodb_uri,
        settings.anthropic_model,
    ) == ("large", 0.85, "mongodb://example:27017", "claude-opus-5")


@pytest.mark.parametrize("boundary_value", ["0", "1", "0.0", "1.0"])
def test_threshold_accepts_the_ratio_boundaries(
    monkeypatch: pytest.MonkeyPatch, boundary_value: str
) -> None:
    monkeypatch.setenv("CONFIDENCE_THRESHOLD", boundary_value)

    settings = load_settings()

    assert settings.confidence_threshold == float(boundary_value)


@pytest.mark.parametrize("invalid_value", ["", "high", "0.7.1"])
def test_non_numeric_threshold_raises(
    monkeypatch: pytest.MonkeyPatch, invalid_value: str
) -> None:
    monkeypatch.setenv("CONFIDENCE_THRESHOLD", invalid_value)

    with pytest.raises(ValueError, match="must be a number"):
        load_settings()


@pytest.mark.parametrize("out_of_range_value", ["-0.1", "1.5", "70"])
def test_threshold_outside_the_ratio_range_raises(
    monkeypatch: pytest.MonkeyPatch, out_of_range_value: str
) -> None:
    monkeypatch.setenv("CONFIDENCE_THRESHOLD", out_of_range_value)

    with pytest.raises(ValueError, match="between 0.0 and 1.0"):
        load_settings()


@pytest.fixture
def isolated_environ(monkeypatch: pytest.MonkeyPatch) -> None:
    """Give the test its own copy of ``os.environ``.

    ``load_dotenv`` writes into ``os.environ`` directly rather than through
    monkeypatch, so without this the values would leak into later tests.
    """
    monkeypatch.setattr(os, "environ", os.environ.copy())


def test_env_file_value_is_used_when_the_variable_is_unset(
    isolated_environ: None, tmp_path: Path
) -> None:
    # Not the default, so a run that never reads the file cannot pass this.
    (tmp_path / ".env").write_text("WHISPER_MODEL_SIZE=large\n", encoding="utf-8")

    load_env_files(tmp_path)

    assert load_settings().whisper_model_size == "large"


def test_exported_variable_wins_over_the_env_file(
    isolated_environ: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / ".env").write_text("WHISPER_MODEL_SIZE=medium\n", encoding="utf-8")
    monkeypatch.setenv("WHISPER_MODEL_SIZE", "tiny")

    load_env_files(tmp_path)

    assert load_settings().whisper_model_size == "tiny"


def test_env_local_takes_precedence_over_env(
    isolated_environ: None, tmp_path: Path
) -> None:
    (tmp_path / ".env").write_text("WHISPER_MODEL_SIZE=medium\n", encoding="utf-8")
    (tmp_path / ".env.local").write_text("WHISPER_MODEL_SIZE=large\n", encoding="utf-8")

    load_env_files(tmp_path)

    assert load_settings().whisper_model_size == "large"


def test_quoted_env_file_value_is_unquoted(
    isolated_environ: None, tmp_path: Path
) -> None:
    # Not the default, so a failed unquote falling back to it cannot pass this.
    (tmp_path / ".env").write_text('WHISPER_MODEL_SIZE="large"\n', encoding="utf-8")

    load_env_files(tmp_path)

    assert load_settings().whisper_model_size == "large"


def test_missing_env_files_are_ignored(isolated_environ: None, tmp_path: Path) -> None:
    load_env_files(tmp_path)

    assert load_settings().whisper_model_size == DEFAULT_WHISPER_MODEL_SIZE


def test_settings_are_frozen() -> None:
    settings = load_settings()

    with pytest.raises(AttributeError):
        settings.whisper_model_size = "tiny"  # type: ignore[misc]
