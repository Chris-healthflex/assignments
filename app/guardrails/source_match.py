from rapidfuzz import fuzz
import re
from app.core.config import settings

# Common medical units that are short and often misspoken/transcribed inconsistently.
COMMON_MEDICAL_UNITS = {
    "mm", "cm", "m", "kg", "g", "mg", "degrees", "degree", "deg",
    "seconds", "sec", "s", "minutes", "min", "hours", "hr", "h",
    "bpm", "mmhg", "ml", "l", "percent", "%", "lbs", "pounds", "kg/m2"
}

_NUM_WORDS = {"0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
              "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine"}

def normalize_text(text: str) -> str:
    text = text.lower()
    for num, word in _NUM_WORDS.items():
        text = re.sub(rf'\b{num}\b', word, text)
    text = re.sub(r'[^\w\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def fuzzy_source_match(value: str, transcript: str) -> int:
    if not value or not transcript:
        return 0
    norm_value = normalize_text(value)
    norm_transcript = normalize_text(transcript)
    if not norm_value:
        return 0
    if norm_value in norm_transcript:
        return 100
    score = fuzz.partial_ratio(norm_value, norm_transcript)
    # Check individual words if value has multiple tokens
    for word in norm_value.split():
        word_score = fuzz.partial_ratio(word, norm_transcript)
        if word_score > score:
            score = word_score
    return score

def adjust_confidence(value: str, original_confidence: float, transcript: str) -> float:
    """Adjust confidence based on fuzzy source match, with special handling for units."""
    if not value:
        return original_confidence

    norm_value = normalize_text(value)

    # If the value is a common medical unit, keep confidence high.
    if norm_value in COMMON_MEDICAL_UNITS:
        return original_confidence

    score = fuzzy_source_match(value, transcript)
    if score >= settings.FUZZY_SOURCE_MATCH_THRESHOLD:
        return original_confidence
    elif score >= settings.FUZZY_PARTIAL_MATCH_THRESHOLD:
        # Soft penalty (was -0.2, now -0.1)
        return max(0.0, original_confidence - 0.1)
    else:
        # Lower penalty (was -0.4, now -0.2)
        return max(0.0, original_confidence - 0.2)