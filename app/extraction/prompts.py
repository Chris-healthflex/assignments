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


SUBJECTIVE = f"""Read the transcript and list what the patient reported and what \
the clinician observed on examination, as distinct from measured numbers.

Produce one entry per distinct finding. For each entry:

testName - What was observed or assessed? For example "Pain", "Surgical scar", \
"Patellar mobility".

conclusion - What was found? Use the transcript's wording.

Numeric range-of-motion measurements do not belong here; they are recorded \
separately. If the transcript contains no such findings, return an empty list.

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


GOALS = f"""Read the transcript and list the treatment goals discussed.

subjectiveGoals - Goals described only in words. Put the description in \
goalDetails.

objectiveGoals - Goals that have a measurable target. Give goalName, \
goalCategory, unitName and the target value.

targetDate - Fill this in ONLY if the transcript states a specific date or \
deadline for that goal. This is the field most often filled in wrongly. If no \
date is spoken anywhere in the transcript, every targetDate must be "".

Do not invent a target number for a goal that was described only in words - \
record it as a subjective goal instead. If no goals are discussed, return \
empty lists.

{_NO_ECHO}"""


PLAN = f"""Read the transcript and record the treatment plan.

recommendation - One entry per recommended treatment. Give sessionType, for \
example "Physiotherapy", and sessionFrequency exactly as stated, for example \
"once weekly for four sessions".

patientAdvice.adviceDetails - What was the patient told to do themselves, \
away from the clinic? For example home exercises, activity restrictions, or \
self-care instructions.

Be careful to distinguish these two. A list of things the therapist will work \
on during treatment is a recommendation, not patient advice. If the \
transcript describes only what the clinician will do and gives the patient no \
separate instructions to follow, adviceDetails must be "".

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
