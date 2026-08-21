import re
from app.guardrails.source_match import fuzzy_source_match

def is_numeric(value: str) -> bool:
    """Check if value looks numeric (e.g., '45', '45.5', '45 degrees')."""
    if not value:
        return False
    # Strip common units
    cleaned = re.sub(r'[^\d.\-]', '', value)
    if not cleaned:
        return False
    try:
        float(cleaned)
        return True
    except ValueError:
        return False

def numeric_source_supported(value: str, transcript: str) -> bool:
    """Check if a numeric value is supported by transcript."""
    # Look for the number itself or nearby context
    num_match = re.search(r'[\d.]+', value)
    if num_match:
        num = num_match.group()
        # Check if number appears in transcript (allow some fuzzy)
        if num in transcript:
            return True
        # Also check as word
        from app.guardrails.source_match import normalize_text
        return normalize_text(num) in normalize_text(transcript)
    return False