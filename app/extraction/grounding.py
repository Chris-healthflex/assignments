"""Deterministic grounding checks - the enforcement of requirement S6.

The brief says the pipeline must "never hallucinate clinical values, scores, or
dates". Prompt wording alone cannot guarantee that, so every value the model
produces is verified against the transcript here, with no LLM involved.

A value is grounded only if all three hold:

1. **Numbers**   every number it contains also appears in the transcript.
2. **Dates**     every date-like token also appears in the transcript.
3. **Lexis**     enough of its content words appear in the transcript that it
                 is a reading of the audio rather than a plausible invention.

A value that fails is *cleared to the empty string and flagged*, never kept.
That is the deliberate trade: a blank flagged field is safe for a clinician to
fill in, whereas a confident wrong measurement is not.

Note on evidence quotes: an earlier design asked the model to supply a
verbatim quote per field. That was dropped because a model that will invent a
measurement will equally invent a quote to support it, and it doubled the
output tokens on a 3B model. Verifying values directly against the transcript
trusts the model with nothing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

#: Spoken numbers must be comparable with written ones. The recording says
#: "eight months" while a correct extraction may say "8 months"; without this
#: mapping that true positive would be rejected as ungrounded.
_WORD_NUMBERS: dict[str, str] = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13",
    "fourteen": "14", "fifteen": "15", "sixteen": "16", "seventeen": "17",
    "eighteen": "18", "nineteen": "19", "twenty": "20", "thirty": "30",
    "forty": "40", "fifty": "50", "sixty": "60", "seventy": "70",
    "eighty": "80", "ninety": "90", "hundred": "100",
    "first": "1", "second": "2", "third": "3", "fourth": "4", "fifth": "5",
    "sixth": "6", "seventh": "7", "eighth": "8", "ninth": "9", "tenth": "10",
    "once": "1", "twice": "2",
    "half": "0.5",
}

_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")

#: Date-shaped tokens. Any of these in a value must be traceable to the audio;
#: the supplied recording contains none, so an emitted date is always invented.
_DATE_PATTERNS = [
    re.compile(r"\b\d{4}-\d{1,2}-\d{1,2}\b"),                 # 2026-09-01
    re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"),         # 01/09/2026
    re.compile(
        r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{1,2}\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{4}\b",
        re.IGNORECASE,
    ),
]

#: Function words carry no evidential weight, so they are excluded from the
#: overlap ratio. Including them would let an invented sentence score highly
#: on "the", "of", "and" alone.
_STOPWORDS = frozenset("""
a an and are as at be been being but by for from had has have he her his how
in into is it its of on or she that the their them then there these they this
to was were what when where which who will with would you your not no do does
did can could should may might must i we us our been also than too very
patient patients reports reported noted show showed showing during following
""".split())

_TOKEN_RE = re.compile(r"[a-z0-9.]+")


#: Units are written one way and spoken another. The recording renders angles
#: as "124" with a degree sign, so an extracted unitName of "degrees" has no
#: literal support unless the symbol is expanded first. Collapsing these to a
#: canonical form stops correct unit names being discarded as invented.
_UNIT_SYNONYMS: dict[str, str] = {
    "deg": "degrees",
    "degree": "degrees",
    "degs": "degrees",
    "cm": "centimetres",
    "centimeters": "centimetres",
    "centimeter": "centimetres",
    "centimetre": "centimetres",
    "mm": "millimetres",
    "millimeters": "millimetres",
    "kg": "kilograms",
    "kilogram": "kilograms",
    "secs": "seconds",
    "sec": "seconds",
    "mins": "minutes",
    "min": "minutes",
}


def normalise(text: str) -> str:
    """Lowercase, expand units and spoken numbers, and collapse whitespace."""
    lowered = text.lower()
    # Expand rather than strip: "124" carries the same meaning as "124 degrees".
    lowered = lowered.replace("°", " degrees ")
    lowered = re.sub(r"[^\w\s.]", " ", lowered)

    words = lowered.split()
    mapped = [
        _UNIT_SYNONYMS.get(word, _WORD_NUMBERS.get(word, word)) for word in words
    ]
    return " ".join(mapped)


def extract_numbers(text: str) -> set[str]:
    """Every number in the text, normalised so 4.50 and 4.5 compare equal."""
    found = set()
    for raw in _NUMBER_RE.findall(normalise(text)):
        value = float(raw)
        # Integers render without a trailing .0 so "8" matches "8.0".
        found.add(str(int(value)) if value.is_integer() else str(value))
    return found


def extract_dates(text: str) -> set[str]:
    return {
        match.group(0).lower()
        for pattern in _DATE_PATTERNS
        for match in pattern.finditer(text)
    }


def content_tokens(text: str) -> list[str]:
    """Meaning-bearing tokens, used for the lexical overlap ratio."""
    return [
        token.strip(".")
        for token in _TOKEN_RE.findall(normalise(text))
        if token.strip(".") and token.strip(".") not in _STOPWORDS
    ]


@dataclass
class Verdict:
    """Outcome of grounding one value against the transcript."""

    grounded: bool
    overlap: float = 0.0
    reasons: list[str] = field(default_factory=list)

    @property
    def reason(self) -> str:
        return "; ".join(self.reasons)


def verify_value(value: str, transcript: str, *, min_overlap: float = 0.5) -> Verdict:
    """Check one extracted value against the transcript.

    Blank values are trivially grounded: "not stated" is the correct answer for
    anything the recording does not cover, and must not be reported as a
    hallucination.
    """
    text = (value or "").strip()
    if not text:
        return Verdict(grounded=True, overlap=1.0)

    reasons: list[str] = []

    # 1. Numbers - the brief's "clinical values, scores".
    transcript_numbers = extract_numbers(transcript)
    invented_numbers = extract_numbers(text) - transcript_numbers
    if invented_numbers:
        reasons.append(
            "number(s) not in transcript: " + ", ".join(sorted(invented_numbers))
        )

    # 2. Dates - called out separately by the brief, and the most tempting
    #    field for a model to fill in helpfully.
    transcript_dates = extract_dates(transcript)
    invented_dates = extract_dates(text) - transcript_dates
    if invented_dates:
        reasons.append("date(s) not in transcript: " + ", ".join(sorted(invented_dates)))

    # 3. Lexical overlap - catches fluent invention that happens to avoid
    #    numbers, e.g. a plausible history nobody mentioned.
    tokens = content_tokens(text)
    overlap = 1.0
    if tokens:
        transcript_tokens = set(content_tokens(transcript))
        overlap = sum(token in transcript_tokens for token in tokens) / len(tokens)
        if overlap < min_overlap:
            reasons.append(f"only {overlap:.0%} of content words appear in the transcript")

    return Verdict(grounded=not reasons, overlap=round(overlap, 3), reasons=reasons)


@dataclass
class GroundingIssue:
    """A value that was rejected, retained for the S5 flag report."""

    path: str
    value: str
    reason: str
    overlap: float


def ground_payload(
    payload: dict, transcript: str, *, prefix: str = "", min_overlap: float = 0.5
) -> tuple[dict, list[GroundingIssue]]:
    """Recursively verify a raw extraction dict, clearing what fails.

    Runs on the plain dict before it becomes a FirstAssessment, so a rejected
    value never exists inside a validated assessment even briefly.
    """
    issues: list[GroundingIssue] = []
    cleaned: dict = {}

    for key, value in payload.items():
        path = f"{prefix}{key}"

        if isinstance(value, dict):
            sub, sub_issues = ground_payload(
                value, transcript, prefix=f"{path}.", min_overlap=min_overlap
            )
            cleaned[key] = sub
            issues.extend(sub_issues)

        elif isinstance(value, list):
            items = []
            for index, item in enumerate(value):
                if isinstance(item, dict):
                    sub, sub_issues = ground_payload(
                        item,
                        transcript,
                        prefix=f"{path}[{index}].",
                        min_overlap=min_overlap,
                    )
                    items.append(sub)
                    issues.extend(sub_issues)
                else:
                    items.append(item)
            cleaned[key] = items

        else:
            text = "" if value is None else str(value)
            verdict = verify_value(text, transcript, min_overlap=min_overlap)
            if verdict.grounded:
                cleaned[key] = text
            else:
                # Clear rather than keep: a blank flagged field is safe for a
                # clinician to complete, a confident wrong value is not.
                cleaned[key] = ""
                issues.append(
                    GroundingIssue(
                        path=path,
                        value=text,
                        reason=verdict.reason,
                        overlap=verdict.overlap,
                    )
                )

    return cleaned, issues
