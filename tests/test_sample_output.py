import json
from pathlib import Path

from app.schemas.assessment import FirstAssessment

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def test_sample_output_validates_against_schema():
    sample_path = DATA_DIR / "sample_output.json"
    assert sample_path.exists(), "data/sample_output.json is missing"

    raw = json.loads(sample_path.read_text(encoding="utf-8"))
    assessment = FirstAssessment(**raw)

    # Round-trips cleanly with no data loss/mutation.
    assert assessment.model_dump() == raw


def test_sample_output_has_no_fabricated_looking_placeholders():
    """Sanity check: none of the classic 'model made something up' smells
    (e.g. lorem-ipsum-style placeholders) appear in the committed sample."""
    sample_path = DATA_DIR / "sample_output.json"
    raw_text = sample_path.read_text(encoding="utf-8").lower()

    suspicious_placeholders = ["lorem ipsum", "n/a", "unknown", "todo", "xxx"]
    for placeholder in suspicious_placeholders:
        assert placeholder not in raw_text, f"Suspicious placeholder found: {placeholder!r}"


def test_sample_transcript_exists_and_is_nonempty():
    transcript_path = DATA_DIR / "sample_transcript.txt"
    assert transcript_path.exists(), "data/sample_transcript.txt is missing"
    assert len(transcript_path.read_text(encoding="utf-8").strip()) > 100
