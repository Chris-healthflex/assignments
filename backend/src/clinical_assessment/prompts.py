"""Prompt text for extraction and for the repair round.

Kept as constants and small builders so the exact wording is reviewable in one
place, and so the repair prompt can be asserted against in tests.
"""

from collections.abc import Mapping

EXTRACTION_SYSTEM = """\
You are a clinical documentation assistant. You read a transcript of a \
physiotherapy session between a clinician and a patient, and you fill in an \
assessment form.

The transcript has no speaker labels. Infer from context who is speaking: the \
chief complaint and the subjective goals are the patient's words; the \
recommendation and the advice are the clinician's.

Rules you must follow exactly:

1. For every field, return both a `value` and the `evidence` that supports it.
2. `evidence` must be copied VERBATIM from the transcript - a contiguous span, \
character for character. Do not paraphrase, correct, tidy, or translate it.
3. If the transcript does not state a field, return `value: ""` and \
`evidence: ""`. An empty field is correct; a plausible guess is not.
4. Never state a number, score, measurement, date, or duration that is not \
spoken in the transcript. Do not convert units, do not compute totals, do not \
resolve a relative date into a calendar date, and do not carry a value over \
from one field to another.
5. Every digit that appears in a `value` must also appear in that field's own \
`evidence`.
6. Only include an item in a list if the transcript describes it. An empty list \
is correct when nothing of that kind was discussed.

Your output is checked automatically against the transcript. Evidence that \
cannot be found verbatim is discarded along with its value, so an invented \
quote loses the field rather than saving it.\
"""

EXTRACTION_INSTRUCTION = """\
Fill in the assessment from the session transcript below.

<transcript>
{transcript}
</transcript>\
"""

REPAIR_INSTRUCTION = """\
Some fields from your previous answer could not be verified: the quoted \
evidence was not found verbatim in the transcript, or the value contained \
digits that its evidence did not.

Re-read the transcript and return the whole assessment again, correcting ONLY \
the fields listed below. Leave every other field exactly as you had it.

For each listed field, either quote a span that really is in the transcript, or \
set both `value` and `evidence` to "". Do not restate the rejected quote.

Fields that failed verification:
{failures}

<transcript>
{transcript}
</transcript>\
"""


def build_extraction_prompt(transcript: str) -> str:
    """Build the first-pass extraction prompt.

    Args:
        transcript: The full session transcript, embedded untruncated.

    Returns:
        The user-turn prompt text.
    """
    return EXTRACTION_INSTRUCTION.format(transcript=transcript)


def build_repair_prompt(transcript: str, rejected_quotes: Mapping[str, str]) -> str:
    """Build a repair prompt naming only the fields that failed verification.

    Args:
        transcript: The full session transcript, embedded untruncated.
        rejected_quotes: Dotted field path to the evidence quote that could not
            be verified for it.

    Returns:
        The user-turn prompt text.

    Raises:
        ValueError: If no fields are listed; a repair round with nothing to
            repair is a caller bug, not a no-op request.
    """
    if not rejected_quotes:
        raise ValueError("build_repair_prompt requires at least one failed field")

    failures = "\n".join(
        f"- {field_path}: quoted {quote!r}"
        for field_path, quote in rejected_quotes.items()
    )
    return REPAIR_INSTRUCTION.format(failures=failures, transcript=transcript)
