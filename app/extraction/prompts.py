"""Prompts for each extraction node.

Design: one focused prompt per clinical section rather than one giant prompt.
Every prompt hammers the same non-negotiable rule — extract ONLY what is stated,
return "" / [] for anything absent, never invent values, scores, or dates.
Output is strict JSON so it parses deterministically.
"""
from __future__ import annotations

SYSTEM = (
    "You are a clinical scribe that extracts structured data from a physiotherapy "
    "assessment transcript. Follow these rules without exception:\n"
    "1. Extract ONLY information explicitly stated in the transcript.\n"
    "2. NEVER invent, infer, or estimate clinical values, measurements, scores, or dates.\n"
    "3. If a field is not stated, return an empty string \"\" (or empty list []).\n"
    "4. Copy numeric values exactly as spoken; do not round or convert units.\n"
    "5. Respond with STRICT JSON only — no prose, no markdown, no code fences."
)

CLINICAL_DETAILS = """From the transcript, extract clinical details.
Return JSON: {{"clinicalHistory": "", "chiefComplaint": "", "duration": ""}}
- clinicalHistory: mechanism of injury, diagnosis, surgery, post-op course.
- chiefComplaint: the patient's presenting complaints in their own terms.
- duration: how long since onset/injury (e.g. "eight months"). "" if not stated.

TRANSCRIPT:
{transcript}
"""

SUBJECTIVE = """From the transcript, extract subjective assessment findings — what was
observed or reported qualitatively (scars, pain on movement, restrictions, swelling,
patient-reported symptoms). These are NOT numeric measurements.
Return JSON: {{"items": [{{"testName": "", "conclusion": ""}}]}}
- testName: the finding/observation.
- conclusion: interpretation if explicitly stated, else "".
Return {{"items": []}} if none stated.

TRANSCRIPT:
{transcript}
"""

OBJECTIVE = """From the transcript, extract objective measurements (range of motion, angles,
strength) as a table. Each distinct measurement is ONE row. When a value is given for
left vs right, put them in the same row's "left" and "right" fields — do NOT create
separate rows per side. Copy numbers exactly (e.g. "124", "4.5"); drop the degree symbol.
Return JSON: {{"tests": [{{"testName": "", "unitName": "", "value": "", "left": "", "right": "", "comments": ""}}]}}
- testName: what was measured (e.g. "Knee flexion", "Ankle dorsiflexion").
- left / right: the measured value for that side. Use "value" only for non-lateral single values.
- unitName: unit if stated (e.g. "degrees"), else "".
Return {{"tests": []}} if none stated.

TRANSCRIPT:
{transcript}
"""

GOALS = """From the transcript, extract treatment goals.
Return JSON with two lists:
{{"subjectiveGoals": [{{"goalDetails": "", "targetDate": ""}}],
  "objectiveGoals": [{{"goalName": "", "goalCategory": "", "unitName": "", "value": "", "targetDate": ""}}]}}
- objectiveGoals: specific rehab targets stated by the clinician (e.g. "restoring extension",
  "strengthening the quadriceps"). goalName is the target; leave category/unit/value/targetDate
  "" unless explicitly stated.
- subjectiveGoals: general/qualitative goals. Usually none — return [] if not stated.
NEVER invent target dates.

TRANSCRIPT:
{transcript}
"""

PLAN = """From the transcript, extract the treatment plan and patient advice.
Return JSON:
{{"recommendation": [{{"sessionType": "", "sessionFrequency": ""}}],
  "patientAdvice": {{"adviceDetails": ""}}}}
- recommendation: therapy recommended + its frequency (e.g. "Physiotherapy", "once weekly for four sessions").
- patientAdvice: home advice/precautions given to the patient; "" if none stated.

TRANSCRIPT:
{transcript}
"""
