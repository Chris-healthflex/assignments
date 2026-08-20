"""Pydantic models for the FirstAssessment contract.

`FirstAssessment` is a *contract*, not a domain model: it must serialise to
exactly the JSON the Stance Health clinician frontend consumes -- same keys,
same nesting, same types. Three rules from the brief drive every choice here.

1. **No extra fields, no renamed keys.** ``extra="forbid"`` on every model
   turns that from a convention into a test: a renamed key arrives as an
   unknown key and fails loudly instead of writing junk to Mongo.
2. **Arrays stay arrays, even with one item.** Every list defaults to ``[]``
   and a ``None`` is normalised to ``[]`` rather than dropped or left null.
3. **String fields are strings, never null.** Every leaf is a ``CleanStr``,
   which defaults to ``""`` and maps ``None`` to ``""`` at the boundary.

**Why camelCase attribute names instead of ``alias_generator=to_camel``:** with
aliases, a plain ``model_dump()`` silently emits snake_case and only
``model_dump(by_alias=True)`` is correct. One forgotten flag anywhere in the
pipeline writes a wrong-shaped document. Declaring the fields in camelCase
makes the correct output the *only* possible output, at the cost of some
non-PEP8 attribute names -- a trade worth making for an exact-match contract.

**Where provenance lives:** confidence, unresolved fields, transcript, ids and
timestamps are all *extra fields* as far as the brief is concerned, so they are
deliberately absent from `FirstAssessment`. They live in `ExtractionFlags` and
`StoredAssessment`, which wrap the assessment rather than polluting it. That is
what lets the parse endpoint return a 422 with field-level detail while the
persisted `assessment` sub-document stays exactly on contract.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated, Any

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field

# The seven top-level sections of the contract, in order. Exported so the tests
# and the extraction agent share one definition of "the schema" instead of two.
SECTIONS: tuple[str, ...] = (
    "clinicalDetails",
    "subjectiveAssessments",
    "objectiveAssessment",
    "subjectiveGoals",
    "objectiveGoals",
    "recommendation",
    "patientAdvice",
)


# How many words either side of a quote count as its context. Wide enough to
# catch the word that changes the meaning ("negative" before a measurement),
# narrow enough not to drag in half the sentence.
CONTEXT_WINDOW = 3

# Below this, a word is not "heard poorly" -- it is a hole in the transcript.
# Ordinary speech dips to 0.5 or 0.6 constantly and that must not raise alarms,
# so only genuinely destroyed audio next to a value drags the value down.
GARBLED = 0.25

# Function words are excluded from the context check. They are the words Whisper
# mangles most often and the ones that carry least clinical meaning: a misheard
# "and" between two goals puts neither of them in doubt, whereas a misheard
# "negative" before a measurement changes what the measurement says. Ignoring
# them removes most of the context check's false alarms without blunting it
# where it matters.
CONTEXT_IGNORED = frozenset(
    """a an and the of to on in with for by at as or but is are was were be been
    it its this that these those from into during""".split()
)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# --------------------------------------------------------------------------- #
# Boundary normalisation
# --------------------------------------------------------------------------- #
def _clean_str(value: Any) -> Any:
    """Normalise a leaf to a string. ``None`` becomes ``""``.

    Rather than trusting every producer to remember rule 3, we enforce it here.
    Numbers are stringified because a clinical value ("120", "45") frequently
    comes back from the LLM as a JSON number. Anything structural -- a dict or
    a list where a string belongs -- is a genuine shape error and is passed
    through untouched so Pydantic rejects it.
    """
    if value is None:
        return ""
    if isinstance(value, bool):  # bool is an int subclass, so check it first
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    return value


def _clean_list(value: Any) -> Any:
    """``None`` becomes ``[]`` so an absent array is still an array."""
    return [] if value is None else value


def _clean_obj(value: Any) -> Any:
    """``None`` becomes ``{}`` so an absent section still renders its keys."""
    return {} if value is None else value


def _bare(token: str) -> str:
    """Lowercase a token and drop the punctuation Whisper hangs off words."""
    return token.lower().strip(" .,;:!?-\"'()")


CleanStr = Annotated[str, BeforeValidator(_clean_str)]
CleanStrList = Annotated[list[CleanStr], BeforeValidator(_clean_list)]


class ContractModel(BaseModel):
    """Base for every model on the wire.

    ``validate_assignment`` matters as much as ``extra="forbid"`` here: it means
    a later ``obj.field = None`` is normalised to ``""`` too, so the invariants
    hold for the whole lifetime of the object, not only at construction.
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
    )


# --------------------------------------------------------------------------- #
# The seven sections
# --------------------------------------------------------------------------- #
class ClinicalDetails(ContractModel):
    clinicalHistory: CleanStr = ""
    chiefComplaint: CleanStr = ""
    duration: CleanStr = ""


class SubjectiveAssessment(ContractModel):
    testName: CleanStr = ""
    conclusion: CleanStr = ""


class ObjectiveTest(ContractModel):
    testName: CleanStr = ""
    unitName: CleanStr = ""
    value: CleanStr = ""
    left: CleanStr = ""
    right: CleanStr = ""
    comments: CleanStr = ""


class ObjectiveAssessment(ContractModel):
    tests: Annotated[list[ObjectiveTest], BeforeValidator(_clean_list)] = Field(
        default_factory=list
    )


class SubjectiveGoal(ContractModel):
    goalDetails: CleanStr = ""
    # Kept as a string on purpose: a transcript says "in about six weeks" far
    # more often than it states a date. Coercing that into an ISO date would be
    # inventing a clinical date, which the brief explicitly forbids.
    targetDate: CleanStr = ""


class ObjectiveGoal(ContractModel):
    goalName: CleanStr = ""
    goalCategory: CleanStr = ""
    unitName: CleanStr = ""
    value: CleanStr = ""
    targetDate: CleanStr = ""


class Recommendation(ContractModel):
    sessionType: CleanStr = ""
    sessionFrequency: CleanStr = ""


class PatientAdvice(ContractModel):
    adviceDetails: CleanStr = ""


# --------------------------------------------------------------------------- #
# Root contract
# --------------------------------------------------------------------------- #
class FirstAssessment(ContractModel):
    """The exact JSON consumed by the clinician frontend. Nothing else.

    Note that ``recommendation`` is singular but holds an array -- that is the
    frontend's spelling, and "fixing" it to ``recommendations`` would be a
    renamed key, which the brief forbids.
    """

    clinicalDetails: Annotated[ClinicalDetails, BeforeValidator(_clean_obj)] = Field(
        default_factory=ClinicalDetails
    )
    subjectiveAssessments: Annotated[
        list[SubjectiveAssessment], BeforeValidator(_clean_list)
    ] = Field(default_factory=list)
    objectiveAssessment: Annotated[ObjectiveAssessment, BeforeValidator(_clean_obj)] = (
        Field(default_factory=ObjectiveAssessment)
    )
    subjectiveGoals: Annotated[list[SubjectiveGoal], BeforeValidator(_clean_list)] = (
        Field(default_factory=list)
    )
    objectiveGoals: Annotated[list[ObjectiveGoal], BeforeValidator(_clean_list)] = Field(
        default_factory=list
    )
    recommendation: Annotated[list[Recommendation], BeforeValidator(_clean_list)] = (
        Field(default_factory=list)
    )
    patientAdvice: Annotated[PatientAdvice, BeforeValidator(_clean_obj)] = Field(
        default_factory=PatientAdvice
    )


# --------------------------------------------------------------------------- #
# Out-of-band provenance -- everything the contract has no room for
# --------------------------------------------------------------------------- #
class FieldEvidence(ContractModel):
    """Why we believe one extracted field, recorded per field rather than once.

    An audio pipeline has two independent ways to be wrong, so one score for the
    whole document cannot express the interesting failure:

    * **Whisper mishears.** "forty degrees" becomes "fourteen degrees". The
      agent then extracts "fourteen" *correctly and confidently* -- it did its
      job on the text it was given, and the number is still wrong.
    * **The agent misreads correct text.** Puts the left-side measurement in
      ``right``, or invents a value that was never discussed.

    ``modelConfidence`` alone catches neither reliably, because self-reported
    LLM confidence is poorly calibrated -- ask for thirty scores and you get
    thirty 0.9s. So it is the weakest of the three signals here and is kept
    mainly for transparency. The two that carry real weight:

    * ``evidenceFound`` -- we require the agent to quote the transcript span it
      took the value from, then verify that span really occurs in the
      transcript **in our own code**. A quote that is not there means the value
      was invented, and that is a mechanical check, not a judgement call.
    * ``audioConfidence`` -- Whisper's own probability for the words in that
      span. This is the only signal that sees the misheard-number case.
    """

    field: CleanStr = ""  # dotted path, e.g. "objectiveAssessment.tests[0].value"
    value: CleanStr = ""  # what we actually wrote into the contract
    evidence: CleanStr = ""  # verbatim transcript span the agent claims as source
    evidenceFound: bool = False  # verified by us against the transcript, not trusted
    modelConfidence: float = Field(default=0.0, ge=0.0, le=1.0)
    # None means "we could not locate the span", which is genuinely different
    # from "Whisper was completely unsure" -- hence nullable here. The
    # never-null rule is a property of the frontend contract, not of our own
    # diagnostics, where losing that distinction would hide a real failure.
    audioConfidence: float | None = Field(default=None, ge=0.0, le=1.0)
    # The weakest word in *and immediately around* the quote. Guards against a
    # well-behaved model quoting its way around a hole in the transcript.
    contextConfidence: float | None = Field(default=None, ge=0.0, le=1.0)
    reason: CleanStr = ""

    @property
    def confidence(self) -> float:
        """The signals collapsed into one score for this field.

        Grounding is a gate rather than a term: if the quoted span is not in the
        transcript, the value is unsupported no matter how sure the model says
        it is, so the score is zero. Past that gate we take the *minimum* of the
        signals that actually reported, instead of an average or a product -- a
        field is only as trustworthy as its weakest link, and multiplying would
        drag every score toward zero and make the threshold meaningless.

        A signal of exactly zero counts as *not reported* rather than as total
        certainty of being wrong. This matters in practice: models are erratic
        about filling in an optional confidence number, and treating a silent
        model as a screaming one would zero out every well-grounded field in a
        section and bury the genuinely suspect ones in noise.
        """
        if not self.evidenceFound:
            return 0.0
        reported = [
            s for s in (self.modelConfidence, self.audioConfidence) if s is not None and s > 0.0
        ]
        score = min(reported) if reported else 0.0
        # Context is a tripwire, not a term. Folding it into the minimum would
        # punish every value that happens to sit near an ordinary mumble, and
        # the resulting noise would bury the real warnings. It only bites when
        # the audio beside the value is genuinely destroyed -- at which point
        # what the value *means* is in doubt, however cleanly it was quoted.
        if self.contextConfidence is not None and self.contextConfidence < GARBLED:
            return min(score, self.contextConfidence)
        return score


class ExtractionFlags(ContractModel):
    """Confidence metadata that drives the 422 on the parse endpoint.

    Lives beside `FirstAssessment`, never inside it.
    """

    # A descriptive summary of the run, *not* the thing the 422 keys off. The
    # gate is per-field (`below()`), because one bad measurement in an otherwise
    # clean assessment is exactly the case an averaged score hides.
    overallConfidence: float = Field(default=0.0, ge=0.0, le=1.0)
    fields: Annotated[list[FieldEvidence], BeforeValidator(_clean_list)] = Field(
        default_factory=list,
        description="One record per populated contract field.",
    )
    unresolvedFields: CleanStrList = Field(
        default_factory=list,
        description="Contract paths the transcript did not support, left empty on purpose.",
    )
    failedSections: CleanStrList = Field(
        default_factory=list,
        description=(
            "Sections left empty because the call that produces them failed. "
            "Distinct from unresolvedFields, which is empty on purpose."
        ),
    )
    warnings: CleanStrList = Field(default_factory=list)

    @property
    def incomplete(self) -> bool:
        """True when part of the document is missing for a reason of ours.

        An empty section normally means the clinician did not mention it, which
        is a trustworthy answer. This is the other case, and the two must never
        be told apart by guesswork.
        """
        return bool(self.failedSections)

    def below(self, threshold: float) -> list[FieldEvidence]:
        """Fields that fail the bar -- the body of the 422, field by field."""
        return [f for f in self.fields if f.confidence < threshold]

    def ungrounded(self) -> list[FieldEvidence]:
        """Fields whose quoted evidence is not in the transcript: invented."""
        return [f for f in self.fields if not f.evidenceFound]

    @classmethod
    def summarise(
        cls,
        fields: list[FieldEvidence],
        unresolved: list[str] | None = None,
        warnings: list[str] | None = None,
        failed_sections: list[str] | None = None,
    ) -> ExtractionFlags:
        """Build the flags, deriving `overallConfidence` from the fields.

        The mean is the honest summary statistic here; the per-field gate is
        what actually protects the clinician. An extraction that populated
        nothing scores zero rather than a vacuous 1.0.
        """
        scores = [f.confidence for f in fields]
        return cls(
            overallConfidence=sum(scores) / len(scores) if scores else 0.0,
            fields=fields,
            unresolvedFields=unresolved or [],
            failedSections=failed_sections or [],
            warnings=warnings or [],
        )


class StoredAssessment(ContractModel):
    """The MongoDB document envelope.

    The contract sits untouched under ``assessment``; ids, timestamps, the
    source transcript and confidence metadata wrap around it.
    """

    id: CleanStr = ""  # Mongo _id as a string; set on read, never by the model
    createdAt: datetime = Field(default_factory=_utcnow)
    audioFilename: CleanStr = ""
    transcript: CleanStr = ""
    flags: ExtractionFlags = Field(default_factory=ExtractionFlags)
    assessment: FirstAssessment = Field(default_factory=FirstAssessment)


# --------------------------------------------------------------------------- #
# Internal envelopes (not contract-locked)
# --------------------------------------------------------------------------- #
class TranscriptWord(ContractModel):
    """One word with the probability Whisper assigned it.

    This is the raw material for `FieldEvidence.audioConfidence`: once we know
    which span a clinical value came from, we can ask how sure Whisper was
    about *those exact words* rather than about the recording as a whole.
    """

    start: float = 0.0
    end: float = 0.0
    word: CleanStr = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class TranscriptSegment(ContractModel):
    start: float = 0.0
    end: float = 0.0
    text: CleanStr = ""
    # Normalised to a 0-1 probability by the transcription layer; faster-whisper
    # reports an average log-probability, which is not comparable to anything
    # else in the pipeline until it is converted.
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    noSpeechProbability: float = Field(default=0.0, ge=0.0, le=1.0)
    words: Annotated[list[TranscriptWord], BeforeValidator(_clean_list)] = Field(
        default_factory=list
    )


class TranscriptionResult(ContractModel):
    text: CleanStr = ""
    language: CleanStr = ""
    durationSec: float = 0.0
    segments: Annotated[list[TranscriptSegment], BeforeValidator(_clean_list)] = Field(
        default_factory=list
    )

    def _words(self) -> list[TranscriptWord]:
        return [w for s in self.segments for w in s.words]

    def _locate(self, span: str) -> list[tuple[int, int]]:
        """Index ranges where `span` occurs in the flattened word list.

        Punctuation is stripped from both sides or nothing longer than a few
        words ever matches -- the quote carries the commas, the word list does
        not.
        """
        needle = " ".join(span.lower().split())
        words = self._words()
        if not needle or not words:
            return []
        tokens = [_bare(t) for t in needle.split()]
        stripped = [_bare(w.word) for w in words]
        return [
            (start, start + len(tokens))
            for start in range(len(stripped) - len(tokens) + 1)
            if stripped[start : start + len(tokens)] == tokens
        ]

    def confidence_for(self, span: str) -> float | None:
        """How sure Whisper was about the words of `span`.

        Returns ``None`` when the span cannot be located, so the caller can tell
        "no audio signal available" apart from "the audio was bad here".
        """
        words = self._words()
        if not words:
            # No word timestamps: fall back to whichever segment contains the
            # span. Coarser, but still local to the quote rather than global.
            needle = " ".join(span.lower().split())
            hits = [
                s.confidence
                for s in self.segments
                if needle and needle in " ".join(s.text.lower().split())
            ]
            return max(hits) if hits else None

        # Weakest word *within* a span: one badly-heard number is enough to make
        # the whole quote untrustworthy. Best match *across* repeated
        # occurrences: a short quote like "degrees" occurs all over a recording
        # and we cannot tell which one the value came from, so charging it for
        # an unrelated bad occurrence invents a problem.
        scores = [min(w.confidence for w in words[a:b]) for a, b in self._locate(span)]
        return max(scores) if scores else None

    def context_confidence(
        self, span: str, window: int = CONTEXT_WINDOW
    ) -> float | None:
        """The weakest word in and immediately around `span`.

        This exists because a well-behaved model can quote its way *around* a
        problem. The recording says "compared with knee gig 5 degrees on the
        right", where the ruined word is almost certainly "negative". Asked for
        the shortest span that establishes the value, a model correctly quotes
        "5 degrees on the right" -- every word of which Whisper heard at 93% or
        better. The quote passes, and the sign error is invisible.

        Widening the view by a few words catches that. Note this is *not* used
        to grade every field -- a merely mediocre neighbour is normal speech.
        It only bites when the audio nearby is genuinely destroyed; see
        `FieldEvidence.confidence`.
        """
        words = self._words()
        hits = self._locate(span)
        if not words or not hits:
            return None
        scores: list[float] = []
        for a, b in hits:
            around = words[max(0, a - window) : min(len(words), b + window)]
            meaningful = [w for w in around if _bare(w.word) not in CONTEXT_IGNORED]
            if meaningful:
                scores.append(min(w.confidence for w in meaningful))
        return max(scores) if scores else None
