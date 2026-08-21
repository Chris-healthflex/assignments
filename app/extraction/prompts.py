"""Per-section extraction prompts and the lenient models they target.

The agent extracts section by section rather than in one call. A 3B model
holds a small flat schema reliably and a seven-section nested one poorly, and
a failure in one section then cannot take the whole assessment down with it.

These models mirror the FirstAssessment sections but use ``extra="ignore"``
instead of ``extra="forbid"``. A small model occasionally emits a stray key;
dropping it silently is better than burning a repair attempt, and the strict
contract is still enforced when the results are assembled into FirstAssessment.

Prompts are phrased as **questions**, not as ``field: description`` lists.
That is not cosmetic. With a descriptive list the 3B model copied a field's
own description back as its value - clinicalHistory came back as "the history
leading to this presentation, any surgery and by whom...". Grounding caught
and cleared it, but the field was then empty. Asking a question, and saying
explicitly not to repeat the question, produced the full correct history from
the same model and the same transcript.
"""

from __future__ import annotations

from typing import List, NamedTuple

from pydantic import BaseModel, ConfigDict


class _Lenient(BaseModel):
    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)


# --------------------------------------------------------------------------
# Section output models
# --------------------------------------------------------------------------
class ClinicalDetailsOut(_Lenient):
    clinicalHistory: str = ""
    chiefComplaint: str = ""
    duration: str = ""


class SubjectiveAssessmentOut(_Lenient):
    testName: str = ""
    conclusion: str = ""


class SubjectiveOut(_Lenient):
    subjectiveAssessments: List[SubjectiveAssessmentOut] = []


class ObjectiveTestOut(_Lenient):
    testName: str = ""
    unitName: str = ""
    value: str = ""
    left: str = ""
    right: str = ""
    comments: str = ""


class ObjectiveOut(_Lenient):
    tests: List[ObjectiveTestOut] = []


class SubjectiveGoalOut(_Lenient):
    goalDetails: str = ""
    targetDate: str = ""


class ObjectiveGoalOut(_Lenient):
    goalName: str = ""
    goalCategory: str = ""
    unitName: str = ""
    value: str = ""
    targetDate: str = ""


class GoalsOut(_Lenient):
    subjectiveGoals: List[SubjectiveGoalOut] = []
    objectiveGoals: List[ObjectiveGoalOut] = []


class RecommendationOut(_Lenient):
    sessionType: str = ""
    sessionFrequency: str = ""


class PatientAdviceOut(_Lenient):
    adviceDetails: str = ""


class PlanOut(_Lenient):
    recommendation: List[RecommendationOut] = []
    patientAdvice: PatientAdviceOut = PatientAdviceOut()


# --------------------------------------------------------------------------
# Prompts
# --------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a clinical documentation assistant. You convert \
transcripts of clinician-patient sessions into structured records that a \
clinician will review and sign.

These rules override every other instruction:

1. Record ONLY what the transcript explicitly states.
2. If the transcript does not state something, return an empty string "". An \
empty field is correct and expected. It is never an error.
3. Never infer or complete a value from your own clinical knowledge. You are \
transcribing, not diagnosing.
4. Never invent a measurement, score, or date. If no date is spoken, the date \
field is "".
5. Prefer the transcript's own wording over your own phrasing.
6. Never copy the question or instruction text into a value.

A missing field costs a clinician a few seconds. A confidently wrong \
measurement can harm a patient. Always choose the empty string when unsure."""

_NO_ECHO = (
    "Do not copy these questions into your answer. Each value must be a fact "
    'taken from the transcript, or "" if the transcript does not state it.'
)


def _with_transcript(instruction: str, transcript: str) -> str:
    return f'{instruction}\n\nTRANSCRIPT:\n"""\n{transcript}\n"""'


CLINICAL_DETAILS = f"""Read the transcript and answer three questions about the \
patient's clinical background.

clinicalHistory - What events led to this presentation? Include the original \
injury or onset, any surgery and who performed it, and what has happened since.

chiefComplaint - What is the main problem the patient presents with?

duration - How long has the problem been present? Copy the time phrase used, \
for example "three weeks" or "eight months".

{_NO_ECHO}"""


SUBJECTIVE = f"""Read the transcript and extract ONLY qualitative clinical findings.

Produce one entry per distinct qualitative finding.

testName - What was observed or assessed? Examples:
"Pain", "Surgical scar", "Patellar mobility", "Swelling".

conclusion - What was found? Use the transcript's wording.

CRITICAL RULE:
This section MUST NOT contain numerical measurements.

If a finding contains a number together with a measurement unit, it belongs
ONLY in objectiveAssessment, NOT in subjectiveAssessments.

For example, these MUST NOT be included here:
- "Knee flexion: 124 degrees"
- "Knee flexion: 130 degrees"
- "Knee extension: 20 degrees"
- "Hip internal rotation: 45 degrees"
- "Hip external rotation: 60 degrees"
- "Ankle dorsiflexion: 4.5 degrees"

These ARE valid subjective findings:
- Pain -> moderate pain with mild irritability
- Surgical scar -> healed
- Knee flexion -> restricted and painful on over pressure
- Knee extension -> restricted
- Swelling -> present
- Patellar mobility -> good
- Left hip extension -> restricted

If there are no qualitative findings, return an empty list.

{_NO_ECHO}"""


OBJECTIVE = f"""Read the transcript and list the objective measurements recorded \
during examination.

Produce one entry per measurement. For each entry:

testName - What was measured? For example "Knee flexion", "Ankle dorsiflexion".

unitName - What unit was used, such as "degrees"? Use "" if none is stated.

left and right - What value was recorded for each side? Give digits only, \
without the unit. If the transcript states only one side, leave the other "".

value - Use this ONLY when a measurement has no left/right distinction. When \
left and right are given, value must be "".

comments - Any qualifier stated about that measurement, otherwise "".

Copy numbers exactly as spoken. Do not round, convert, or supply a missing \
side. If the transcript contains no measurements, return an empty list.

"Bilaterally" means the SAME value applies to both sides - record it as
left and right, not as one side of a different test. Two measurements
stated back to back, each "bilaterally", are two separate entries.

List EVERY measurement stated. They often arrive as one long sequence -
"... 45 degrees bilaterally, ... 60 degrees bilaterally, ..." - and it is easy
to skip one in the middle. Work through the sequence and account for every
number that carries a unit.

{_NO_ECHO}"""


GOALS = f"""Read the transcript and list ONLY the treatment goals that are
explicitly stated in the transcript.

There are two types of goals:

1. subjectiveGoals:
   Goals described in words without an explicitly stated measurable target.

2. objectiveGoals:
   A goal may be placed here ONLY when the transcript explicitly states
   a measurable target value for that goal.

CRITICAL RULES:

- NEVER infer a measurable target from an examination measurement.
- NEVER convert an existing examination measurement into a goal.
- NEVER invent a target value, unit, category, percentage, or number.
- A goal such as "restore extension" is NOT an objectiveGoal unless the
  transcript explicitly says something like "restore extension to 0 degrees".
- A goal such as "improve knee stability" is a subjectiveGoal unless an
  explicit measurable target is stated.
- A goal such as "strengthen quadriceps" is a subjectiveGoal unless an
  explicit measurable target is stated.
- If a goal has no explicitly stated target value, it MUST be a
  subjectiveGoal.
- If there are NO goals with explicitly stated measurable target values,
  objectiveGoals MUST be [].

IMPORTANT:
Measurements describing the patient's CURRENT condition are NOT treatment
targets.

For example, if the transcript says:
"left knee extension is 20 degrees"

this is a current objective measurement, NOT an objective goal.

Do NOT turn it into:
"restore extension -> 20 degrees"

Similarly, if the transcript says:
"left knee flexion is 124 degrees"

do NOT use 124 degrees as a goal unless the transcript explicitly says
that 124 degrees is the TARGET.

For objectiveGoals:

goalName - The explicitly stated goal.
goalCategory - Fill ONLY if explicitly stated in the transcript.
unitName - Fill ONLY if explicitly stated for the target.
value - The explicitly stated target value.
targetDate - Fill ONLY if an explicit date or deadline is stated for
that specific goal.

For subjectiveGoals:

goalDetails - Record the goal in the transcript's wording.
targetDate - Fill ONLY if an explicit date or deadline is stated.

For this transcript, phrases such as:
"restore extension"
"improve knee stability"
"improve single-leg stability"
"strengthen the quadriceps"
"strengthen functional lower limb musculature"
"improve ankle mobility"
"activate the posterior chain"

must NOT be converted into objectiveGoals merely because they could
theoretically be measured.

If the transcript does not explicitly provide a measurable target,
leave objectiveGoals empty.

If no goals are discussed, return empty lists.

{_NO_ECHO}"""
PLAN = f"""Read the transcript and record the treatment plan exactly as stated.

There are two separate outputs:

1. recommendation
2. patientAdvice

RECOMMENDATION:
Record treatments or activities that the clinician/therapist plans or
recommends as part of the clinical treatment.

For each recommendation:
- sessionType - The type of treatment explicitly stated, for example
  "Physiotherapy".
- sessionFrequency - The frequency explicitly stated, for example
  "once weekly for four sessions".

PATIENT ADVICE:
Record ONLY instructions that the clinician explicitly tells the patient
to do themselves outside the treatment session.

Examples of patient advice:
- "Perform these exercises at home."
- "Avoid prolonged standing."
- "Continue the prescribed home exercise program."
- "Use ice at home."

CRITICAL DISTINCTION:

A treatment goal or something the therapist plans to work on is NOT
patientAdvice.

For example:
"strengthen the quadriceps"
"improve ankle mobility"
"activate the posterior chain"
"improve knee stability"

These describe the treatment/clinical goals. They are NOT patient advice
unless the transcript explicitly says that the patient was instructed to
perform them themselves.

Similarly, a statement such as:
"Physiotherapy will focus on strengthening the quadriceps"

must NOT be placed in patientAdvice.

It belongs to the treatment plan.

Only put something in patientAdvice when the transcript explicitly
indicates that the patient was told or instructed to do it.

If the transcript contains treatment recommendations but NO explicit
instructions for the patient to follow outside treatment, return:

patientAdvice.adviceDetails = ""

NEVER invent patient instructions.

{_NO_ECHO}"""


class SectionSpec(NamedTuple):
    """One extraction node: a key, a label, an output model, and its prompt."""

    key: str
    label: str
    model_cls: type[BaseModel]
    instruction: str


#: Executed in this order. Each becomes one node in the LangGraph agent.
SECTION_SPECS: tuple[SectionSpec, ...] = (
    SectionSpec("clinicalDetails", "clinical details", ClinicalDetailsOut, CLINICAL_DETAILS),
    SectionSpec("subjective", "subjective assessment", SubjectiveOut, SUBJECTIVE),
    SectionSpec("objective", "objective measurements", ObjectiveOut, OBJECTIVE),
    SectionSpec("goals", "goals", GoalsOut, GOALS),
    SectionSpec("plan", "plan and advice", PlanOut, PLAN),
)


def build_user_prompt(spec: SectionSpec, transcript: str) -> str:
    return _with_transcript(spec.instruction, transcript)
