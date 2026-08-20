"""Tests for the D5 pipeline script and the committed sample output.

The sample output is checked into the repository and referenced by the README,
so it is validated here. A committed artifact that no longer matches the schema
would be worse than none at all - a reviewer would take it as the contract.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from app.schemas.first_assessment import FirstAssessment, SECTION_KEYS

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = PROJECT_ROOT / "scripts" / "run_pipeline.py"
SAMPLE = PROJECT_ROOT / "data" / "sample_output.json"


def test_script_exists():
    assert SCRIPT.is_file()


def test_script_help_runs_without_models():
    """--help must work without loading Whisper, Ollama or MongoDB."""
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0
    for flag in ("--file", "--raw", "--save", "--cached", "--out"):
        assert flag in result.stdout


def test_script_reports_a_missing_file_cleanly(tmp_path):
    """A bad path should be a clear message, not a traceback."""
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--file", str(tmp_path / "nope.wav")],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 2
    assert "no such file" in result.stderr.lower()
    assert "Traceback" not in result.stderr


# --------------------------------------------------------------------------
# The committed sample output
# --------------------------------------------------------------------------
@pytest.fixture
def sample() -> dict:
    if not SAMPLE.exists():
        pytest.skip("sample_output.json not generated yet")
    return json.loads(SAMPLE.read_text(encoding="utf-8"))


def test_sample_output_validates_against_the_schema(sample):
    """The committed artifact must still satisfy the contract."""
    assessment = FirstAssessment.model_validate(sample["assessment"])
    assert assessment.model_dump() == sample["assessment"]


def test_sample_output_has_exactly_seven_keys(sample):
    assert list(sample["assessment"]) == list(SECTION_KEYS)


def test_sample_output_contains_no_nulls(sample):
    """Rule 3 holds for real output, not just for constructed test cases."""

    def walk(node, path="assessment"):
        assert node is not None, f"null at {path}"
        if isinstance(node, dict):
            for key, value in node.items():
                walk(value, f"{path}.{key}")
        elif isinstance(node, list):
            for index, value in enumerate(node):
                walk(value, f"{path}[{index}]")

    walk(sample["assessment"])


def test_sample_output_invented_no_dates(sample):
    """S6 on real data: the recording contains no dates, so none may appear.

    This is the strongest evidence in the repository that the anti-
    hallucination guard works, because it is a real transcript rather than a
    constructed fixture.
    """
    goals = sample["assessment"]["objectiveGoals"] + sample["assessment"]["subjectiveGoals"]
    assert goals, "expected the sample to contain goals"
    assert all(goal["targetDate"] == "" for goal in goals)


def test_sample_output_captured_the_real_measurements(sample):
    """Spot-check against the audio: knee flexion was 124 left, 130 right."""
    tests = sample["assessment"]["objectiveAssessment"]["tests"]
    numbers = {
        value
        for test in tests
        for value in (test["value"], test["left"], test["right"])
        if value
    }
    assert {"124", "130"} <= numbers


def test_sample_output_reports_flags_outside_the_assessment(sample):
    """S5 metadata must not be smuggled into the schema."""
    assert "flaggedFields" in sample
    assert "flaggedFields" not in sample["assessment"]
    assert "confidence" not in sample["assessment"]


def test_sample_output_met_the_confidence_threshold(sample):
    confidence = sample["confidence"]
    assert confidence["meetsThreshold"]
    assert confidence["overall"] >= confidence["threshold"]


def test_script_forces_utf8_on_its_streams():
    """Redirected output on Windows otherwise takes the locale encoding.

    Every range-of-motion value carries a degree sign, so under cp1252
    `run_pipeline.py > out.json` wrote bytes that json.load could not read
    back. The committed artifact below is the same content, so it doubles as
    the check that the encoding path holds.
    """
    source = SCRIPT.read_text(encoding="utf-8")
    assert 'reconfigure(encoding="utf-8")' in source


def test_sample_output_is_utf8_with_its_symbols_intact():
    if not SAMPLE.exists():
        pytest.skip("sample_output.json not generated yet")

    raw = SAMPLE.read_text(encoding="utf-8")      # raises if not valid UTF-8
    assert json.loads(raw)
    # The transcript quotes measurements as "124°"; losing the sign silently
    # would mean the encoding regressed.
    assert "\u00b0" in raw
