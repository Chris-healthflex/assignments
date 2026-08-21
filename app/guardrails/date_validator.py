from datetime import datetime, date
from typing import Optional
import re

def is_valid_date_string(value: str) -> bool:
    """Check if value is a valid date or common date phrase."""
    if not value:
        return True  # empty is valid absence
    # Try common formats
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%b-%Y", "%d %B %Y", "%B %d, %Y"):
        try:
            datetime.strptime(value, fmt)
            return True
        except ValueError:
            continue
    # Allow phrases like "2 weeks", "3 months"
    if re.match(r'^\d+\s+(day|days|week|weeks|month|months|year|years)$', value, re.I):
        return True
    # Allow vague dates like "next month"
    if value.lower() in {"next month", "next week", "tomorrow", "today", "in 2 weeks", "asap"}:
        return True
    return False

def date_source_supported(value: str, transcript: str) -> bool:
    """Check if the date value has any support in transcript."""
    # Use a simple fuzzy match for date-like strings
    from app.guardrails.source_match import fuzzy_source_match
    score = fuzzy_source_match(value, transcript)
    return score >= 70