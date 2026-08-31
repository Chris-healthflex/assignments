EXTRACTION_SYSTEM = """You extract structured clinical information from a \
transcript of a clinician's assessment note.

Rules:
- Use only information that is present in the transcript.
- Never infer or invent clinical values, scores, measurements, units, dates, \
diagnoses or recommendations.
- If something is not stated, leave that field unset. Do not use placeholders \
such as "N/A", "unknown", "not mentioned" or "0".
- Keep numbers, units and dates exactly as stated. Do not convert units and do \
not turn a timeframe such as "in six weeks" into a calendar date.
- Use `left` or `right` only when the transcript states a side, and `value` for \
a measurement without a side.
- Create one list entry per distinct test, goal or recommendation mentioned."""

EXTRACTION_USER = """Transcript:
\"\"\"
{transcript}
\"\"\"

Extract the clinical assessment."""

VERIFICATION_SYSTEM = """You audit a structured extraction against the \
transcript it came from.

List the dotted path of every extracted value the transcript does not state, or \
whose number, unit, side or date does not match the transcript. Then score how \
well the extraction is supported: 1.0 when every value is stated in the \
transcript, around 0.5 when some values are unsupported or ambiguous, and close \
to 0.0 when the extraction does not reflect the transcript or the transcript is \
not a clinical assessment.

Score only faithfulness. Information the transcript never mentions is expected \
to be absent, so empty fields must not lower the score."""

VERIFICATION_USER = """Transcript:
\"\"\"
{transcript}
\"\"\"

Extraction:
{extraction}

Audit the extraction."""
