"""Deterministic checks that extracted strings are supported by the transcript."""

import re
from typing import Any

from app.models.first_assessment import FirstAssessment


_NUMBER_WORDS = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "eleven": "11",
    "twelve": "12",
    "thirteen": "13",
    "fourteen": "14",
    "fifteen": "15",
    "sixteen": "16",
    "seventeen": "17",
    "eighteen": "18",
    "nineteen": "19",
    "twenty": "20",
    "thirty": "30",
    "forty": "40",
    "fifty": "50",
    "sixty": "60",
    "seventy": "70",
    "eighty": "80",
    "ninety": "90",
}
_UNIT_WORDS = {"degree": "degrees", "deg": "degrees", "°": "degrees"}
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")
_DATE_RE = re.compile(
    r"\b(?:\d{4}-\d{1,2}-\d{1,2}|\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|"
    r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{1,2}(?:,\s*\d{4})?)\b",
    re.IGNORECASE,
)
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by",
    "for", "from", "had", "has", "have", "he", "her", "his", "in",
    "into", "is", "it", "its", "of", "on", "or", "she", "that", "the",
    "their", "them", "then", "there", "these", "they", "this", "to",
    "was", "were", "with", "would", "patient", "patients",
}


def _normalise(text: str) -> str:
    lowered = text.lower().replace("°", " degrees ")
    lowered = re.sub(r"[^a-z0-9.\s]", " ", lowered)
    words = [_NUMBER_WORDS.get(word, _UNIT_WORDS.get(word, word)) for word in lowered.split()]
    return " ".join(words)


def _numbers(text: str) -> set[str]:
    values: set[str] = set()
    for token in _TOKEN_RE.findall(_normalise(text)):
        if re.fullmatch(r"\d+(?:\.\d+)?", token):
            number = float(token)
            values.add(str(int(number)) if number.is_integer() else str(number))
    return values


def _dates(text: str) -> set[str]:
    return {match.group(0).lower() for match in _DATE_RE.finditer(text)}


def _content_tokens(text: str) -> list[str]:
    return [token for token in _TOKEN_RE.findall(_normalise(text)) if token not in _STOPWORDS]


def _token_supported(token: str, transcript_tokens: set[str]) -> bool:
    if token in transcript_tokens:
        return True
    return any(
        len(token) >= 5
        and len(candidate) >= 5
        and (token.startswith(candidate) or candidate.startswith(token))
        for candidate in transcript_tokens
    )


def verify_value(value: str, transcript: str, min_overlap: float = 0.5) -> tuple[bool, str]:
    """Return whether a non-empty extracted value has transcript support."""
    if not value.strip():
        return True, ""

    missing_dates = _dates(value) - _dates(transcript)
    if missing_dates:
        return False, f"Date(s) not found in transcript: {', '.join(sorted(missing_dates))}"

    missing_numbers = _numbers(value) - _numbers(transcript)
    if missing_numbers:
        return False, f"Number(s) not found in transcript: {', '.join(sorted(missing_numbers))}"

    value_tokens = _content_tokens(value)
    transcript_tokens = set(_content_tokens(transcript))
    if value_tokens:
        overlap = sum(_token_supported(token, transcript_tokens) for token in value_tokens) / len(value_tokens)
        if overlap < min_overlap:
            return False, f"Only {overlap:.0%} of meaningful words appear in transcript"
    return True, ""


def _ground_node(node: Any, transcript: str, path: str, issues: list[dict]) -> Any:
    if isinstance(node, dict):
        return {
            key: _ground_node(value, transcript, f"{path}.{key}" if path else key, issues)
            for key, value in node.items()
        }
    if isinstance(node, list):
        return [_ground_node(value, transcript, f"{path}[{index}]", issues) for index, value in enumerate(node)]
    if isinstance(node, str) and node.strip():
        grounded, reason = verify_value(node, transcript)
        if not grounded:
            issues.append({"field": path, "value": node, "confidence": 0.0, "message": reason})
            return ""
    return node


def ground_assessment(assessment: FirstAssessment, transcript: str) -> tuple[FirstAssessment, list[dict]]:
    """Clear unsupported values while preserving the exact schema shape."""
    issues: list[dict] = []
    payload = _ground_node(assessment.model_dump(), transcript, "", issues)
    return FirstAssessment.model_validate(payload), issues
