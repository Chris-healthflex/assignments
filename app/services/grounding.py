import re
from typing import Optional, Tuple, List

# Number word dictionary for spoken numeric matching
WORD_TO_NUM = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
    "hundred": 100,
}

NUM_TO_WORD = {v: k for k, v in WORD_TO_NUM.items() if v <= 20 or v % 10 == 0}


def number_to_spoken_words(num: int) -> List[str]:
    """Generates possible spoken words for a number (e.g. 52 -> ['52', 'fifty two', 'fifty-two'])."""
    forms = [str(num)]
    if num in NUM_TO_WORD:
        forms.append(NUM_TO_WORD[num])
    elif 20 < num < 100:
        tens = (num // 10) * 10
        units = num % 10
        if tens in NUM_TO_WORD and units in NUM_TO_WORD:
            forms.append(f"{NUM_TO_WORD[tens]} {NUM_TO_WORD[units]}")
            forms.append(f"{NUM_TO_WORD[tens]}-{NUM_TO_WORD[units]}")
    elif 100 <= num < 1000:
        hundreds = num // 100
        remainder = num % 100
        base = f"{NUM_TO_WORD.get(hundreds, str(hundreds))} hundred"
        forms.append(base)
        if remainder > 0:
            sub = number_to_spoken_words(remainder)
            for s in sub:
                forms.append(f"{base} and {s}")
                forms.append(f"{base} {s}")
    return forms


def extract_numbers(text: str) -> List[int]:
    """Extracts all integer numbers from text (both digits and simple word numerals)."""
    found = []
    # Digits
    digits = re.findall(r"\b\d+\b", text)
    for d in digits:
        try:
            found.append(int(d))
        except ValueError:
            pass
    return found


def is_numeric_field(field_path: str) -> bool:
    """Identifies if a dot-path points to a numeric or measurement field."""
    low = field_path.lower()
    return any(term in low for term in ["value", "left", "right", "rom", "pain", "score", "degree"])


def is_date_or_duration_field(field_path: str) -> bool:
    """Identifies if a dot-path points to a date or duration field."""
    low = field_path.lower()
    return any(term in low for term in ["date", "duration", "targetdate"])


def ground_number(
    transcript: str,
    number: int,
    anchor: Optional[str] = None,
    window_chars: int = 250,
) -> Tuple[bool, Optional[str]]:
    """
    Checks if a number appears in the transcript, either as digits or spoken English words.
    If an anchor (e.g. 'flexion', 'extension', test name) is provided, searches in windows around anchor first.
    Returns (grounded, evidence_span).
    """
    candidate_forms = number_to_spoken_words(number)

    # 1. If anchor provided, look in window around anchor
    if anchor:
        anchor_clean = re.escape(anchor.strip().lower())
        for match in re.finditer(anchor_clean, transcript.lower()):
            start = max(0, match.start() - window_chars)
            end = min(len(transcript), match.end() + window_chars)
            window_text = transcript[start:end]

            for form in candidate_forms:
                form_pat = r"\b" + re.escape(form) + r"\b"
                fm = re.search(form_pat, window_text, re.IGNORECASE)
                if fm:
                    # Found in anchor window
                    exact_start = start + fm.start()
                    exact_end = start + fm.end()
                    # Expand slightly for readable context span (up to ~60 chars)
                    ctx_start = max(0, exact_start - 25)
                    ctx_end = min(len(transcript), exact_end + 25)
                    return True, transcript[ctx_start:ctx_end].strip()

    # 2. Otherwise, check across full transcript
    for form in candidate_forms:
        form_pat = r"\b" + re.escape(form) + r"\b"
        fm = re.search(form_pat, transcript, re.IGNORECASE)
        if fm:
            ctx_start = max(0, fm.start() - 25)
            ctx_end = min(len(transcript), fm.end() + 25)
            return True, transcript[ctx_start:ctx_end].strip()

    return False, None


def ground_date_or_duration(
    transcript: str,
    value: str,
    anchor: Optional[str] = None,
) -> Tuple[bool, Optional[str]]:
    """
    Checks if a date or duration phrase (e.g. '6 weeks', 'six weeks', '2 months') has transcript support.
    """
    if not value or not value.strip():
        return False, None

    cleaned_val = value.strip().lower()
    candidate_phrases = [cleaned_val]

    # Handle duration like "6 weeks" -> "six weeks"
    nums = extract_numbers(cleaned_val)
    if nums:
        for n in nums:
            spoken_list = number_to_spoken_words(n)
            for sp in spoken_list:
                candidate_phrases.append(cleaned_val.replace(str(n), sp))

    # Also handle relative patterns like "six weeks", "few days", "months"
    for phrase in candidate_phrases:
        pat = r"\b" + re.escape(phrase) + r"\b"
        fm = re.search(pat, transcript, re.IGNORECASE)
        if fm:
            ctx_start = max(0, fm.start() - 25)
            ctx_end = min(len(transcript), fm.end() + 25)
            return True, transcript[ctx_start:ctx_end].strip()

    # If value is an ISO date like YYYY-MM-DD, search for spoken relative indicators
    if re.match(r"^\d{4}-\d{2}-\d{2}$", cleaned_val):
        # Look for temporal words near anchor
        temporal_words = ["week", "weeks", "month", "months", "days", "next visit", "follow up"]
        for tw in temporal_words:
            fm = re.search(r"\b" + re.escape(tw) + r"\b", transcript, re.IGNORECASE)
            if fm:
                ctx_start = max(0, fm.start() - 25)
                ctx_end = min(len(transcript), fm.end() + 25)
                return True, transcript[ctx_start:ctx_end].strip()

    return False, None


def ground_field(
    transcript: str,
    field_path: str,
    value: str,
    anchor: Optional[str] = None,
) -> Tuple[bool, Optional[str]]:
    """
    Dispatches to numeric or date/duration grounding based on field type.
    For unpopulated / empty fields, returns (True, None) so empty fields aren't flagged as ungrounded.
    """
    if not value or not str(value).strip():
        return True, None

    val_str = str(value).strip()

    if is_numeric_field(field_path):
        nums = extract_numbers(val_str)
        if not nums:
            # Value is text like 'Normal' or 'Positive' or 'Painful'
            # Check if text itself appears in transcript
            fm = re.search(r"\b" + re.escape(val_str.lower()) + r"\b", transcript.lower())
            if fm:
                ctx_start = max(0, fm.start() - 25)
                ctx_end = min(len(transcript), fm.end() + 25)
                return True, transcript[ctx_start:ctx_end].strip()
            return False, None

        # If it has numbers (e.g. 52, or 4/10), all extracted numbers must be grounded
        for n in nums:
            grounded, span = ground_number(transcript, n, anchor=anchor)
            if not grounded:
                return False, None
        # Return evidence span of the last grounded number
        return True, span

    elif is_date_or_duration_field(field_path):
        return ground_date_or_duration(transcript, val_str, anchor=anchor)

    # For free-text fields (chiefComplaint, etc.), check if substantial words match
    words = [w for w in re.findall(r"\b\w{4,}\b", val_str.lower())]
    if words:
        matches = [w for w in words if w in transcript.lower()]
        if len(matches) >= 1:
            fm = re.search(r"\b" + re.escape(matches[0]) + r"\b", transcript, re.IGNORECASE)
            if fm:
                ctx_start = max(0, fm.start() - 25)
                ctx_end = min(len(transcript), fm.end() + 25)
                return True, transcript[ctx_start:ctx_end].strip()

    return False, None
