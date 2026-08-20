"""Tests for the HTTP layer.

Whisper and Gemini are faked here on purpose. What these tests are about is the
boundary (status codes, the shape of the 422, upload limits, route ordering),
and running the real model would make them slow, expensive and non-deterministic
without testing any more of the code that lives in this file. The extraction
itself has its own suite.
"""

from __future__ import annotations

import httpx
import pytest
from pymongo.errors import ServerSelectionTimeoutError

from app import db, main
from app.extraction import ExtractionFailed, ExtractionResult, ExtractionUnavailable
from app.main import app
from app.schemas import (
    SECTIONS,
    ClinicalDetails,
    ExtractionFlags,
    FieldEvidence,
    FirstAssessment,
    ObjectiveTest,
    StoredAssessment,
    TranscriptionResult,
)
from app.transcription import TranscriptionError

WAV_BYTES = b"RIFF$\x00\x00\x00WAVEfmt " + b"\x00" * 32


@pytest.fixture
async def client():
    """Talk to the app in-process. Lifespan is skipped, so no Mongo is needed."""
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #
def make_assessment() -> FirstAssessment:
    assessment = FirstAssessment(
        clinicalDetails=ClinicalDetails(
            clinicalHistory="RTA eight months ago.",
            chiefComplaint="Left knee pain",
            duration="eight months",
        )
    )
    assessment.objectiveAssessment.tests = [
        ObjectiveTest(
            testName="Knee flexion",
            unitName="degrees",
            value="",
            left="124",
            right="130",
            comments="",
        )
    ]
    return assessment


def evidence(field: str, confidence: float, *, found: bool = True) -> FieldEvidence:
    """A field scored at roughly `confidence`, via the audio signal."""
    return FieldEvidence(
        field=field,
        value="124",
        evidence="left knee flexion 124 degrees",
        evidenceFound=found,
        modelConfidence=0.95,
        audioConfidence=confidence,
    )


def install_pipeline(
    monkeypatch, fields: list[FieldEvidence], failed: list[str] | None = None
) -> None:
    """Replace Whisper and the agent with something instant and predictable.

    `failed` names sections the extraction could not produce, as happens when
    one of the three concurrent model calls does not return.
    """
    transcription = TranscriptionResult(
        text="left knee flexion 124 degrees", language="en", durationSec=1.0
    )
    result = ExtractionResult(
        assessment=make_assessment(),
        flags=ExtractionFlags.summarise(fields, failed_sections=failed),
    )
    monkeypatch.setattr(main, "transcribe", lambda *a, **k: transcription)
    monkeypatch.setattr(main, "extract", lambda *a, **k: result)


def install_db(monkeypatch, **overrides) -> None:
    """Stub the database so HTTP behaviour can be tested without one."""

    async def save(stored):
        return "507f1f77bcf86cd799439011"

    async def get(assessment_id):
        return None

    async def listing(day=None, *, limit=50, skip=0):
        return []

    for name, default in (
        ("save_assessment", save),
        ("get_assessment", get),
        ("list_assessments", listing),
    ):
        monkeypatch.setattr(db, name, overrides.get(name, default))


# --------------------------------------------------------------------------- #
# Path translation
# --------------------------------------------------------------------------- #
def test_a_dotted_path_becomes_a_json_pointer_style_loc():
    assert main._loc("objectiveAssessment.tests[1].left") == [
        "assessment",
        "objectiveAssessment",
        "tests",
        1,
        "left",
    ]


def test_a_top_level_path_is_still_prefixed_with_assessment():
    # The loc has to be valid against the body the client is holding, which
    # nests the contract, not against the bare contract.
    assert main._loc("clinicalDetails.duration") == [
        "assessment",
        "clinicalDetails",
        "duration",
    ]


def test_a_bare_array_element_keeps_its_index():
    assert main._loc("recommendation[0].sessionType") == [
        "assessment",
        "recommendation",
        0,
        "sessionType",
    ]


def test_an_untraceable_value_is_a_different_error_than_a_quiet_one():
    # Worth distinguishing: one means the model invented something, the other
    # means the microphone failed. They call for different responses.
    invented = main._field_error(evidence("a.b", 0.9, found=False), 0.6)
    misheard = main._field_error(evidence("a.b", 0.1), 0.6)
    assert invented.type == "unverified_evidence"
    assert misheard.type == "low_confidence"


# --------------------------------------------------------------------------- #
# POST /assessments/parse
# --------------------------------------------------------------------------- #
async def test_a_confident_parse_returns_the_envelope(client, monkeypatch):
    install_pipeline(monkeypatch, [evidence("clinicalDetails.duration", 0.95)])

    response = await client.post(
        "/assessments/parse", files={"file": ("session.wav", WAV_BYTES, "audio/wav")}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["audioFilename"] == "session.wav"
    assert body["transcript"] == "left knee flexion 124 degrees"
    assert body["id"] == ""  # nothing saved yet


async def test_the_returned_contract_has_exactly_the_seven_sections(client, monkeypatch):
    # The exact-match rule, checked at the HTTP boundary rather than trusted to
    # survive serialisation.
    install_pipeline(monkeypatch, [evidence("clinicalDetails.duration", 0.95)])

    response = await client.post(
        "/assessments/parse", files={"file": ("session.wav", WAV_BYTES, "audio/wav")}
    )

    assert tuple(response.json()["assessment"].keys()) == SECTIONS


async def test_a_low_confidence_field_makes_it_a_422(client, monkeypatch):
    install_pipeline(
        monkeypatch,
        [
            evidence("clinicalDetails.duration", 0.95),
            evidence("objectiveAssessment.tests[0].left", 0.05),
        ],
    )

    response = await client.post(
        "/assessments/parse", files={"file": ("session.wav", WAV_BYTES, "audio/wav")}
    )

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert len(detail) == 1  # only the failing field, not the whole run
    assert detail[0]["loc"] == ["assessment", "objectiveAssessment", "tests", 0, "left"]
    assert detail[0]["type"] == "low_confidence"


async def test_the_422_says_what_was_heard_and_how_sure_it_was(client, monkeypatch):
    install_pipeline(monkeypatch, [evidence("objectiveAssessment.tests[0].left", 0.05)])

    response = await client.post(
        "/assessments/parse", files={"file": ("session.wav", WAV_BYTES, "audio/wav")}
    )

    ctx = response.json()["detail"][0]["ctx"]
    assert ctx["value"] == "124"
    assert ctx["evidence"] == "left knee flexion 124 degrees"
    assert ctx["audioConfidence"] == pytest.approx(0.05)
    assert ctx["confidence"] == pytest.approx(0.05)


async def test_the_422_still_carries_the_draft(client, monkeypatch):
    # Withholding it would leave the clinician with an error and nothing to
    # correct. The flagged fields are exactly the ones they need to look at.
    install_pipeline(monkeypatch, [evidence("objectiveAssessment.tests[0].left", 0.05)])

    response = await client.post(
        "/assessments/parse", files={"file": ("session.wav", WAV_BYTES, "audio/wav")}
    )

    body = response.json()
    assert tuple(body["assessment"].keys()) == SECTIONS
    assert body["assessment"]["objectiveAssessment"]["tests"][0]["left"] == "124"
    assert body["transcript"]


async def test_an_extraction_that_found_nothing_is_not_a_422(client, monkeypatch):
    # No fields means nothing failed the bar. An empty transcript is handled
    # upstream; silence in the recording is a legitimate result, not an error.
    install_pipeline(monkeypatch, [])

    response = await client.post(
        "/assessments/parse", files={"file": ("session.wav", WAV_BYTES, "audio/wav")}
    )

    assert response.status_code == 200


async def test_a_transcription_failure_is_a_422_not_a_500(client, monkeypatch):
    install_pipeline(monkeypatch, [])

    def boom(*args, **kwargs):
        raise TranscriptionError("Audio file not found")

    monkeypatch.setattr(main, "transcribe", boom)

    response = await client.post(
        "/assessments/parse", files={"file": ("session.wav", WAV_BYTES, "audio/wav")}
    )

    assert response.status_code == 422
    assert "Audio file not found" in response.json()["detail"]


# --------------------------------------------------------------------------- #
# Upload validation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("filename", ["session.wav", "session.mp3", "session.m4a", "session.flac"])
async def test_the_common_audio_containers_are_accepted(client, monkeypatch, filename):
    # faster-whisper decodes these through its bundled PyAV, with no separate
    # ffmpeg install. Transcription is faked here; the real decode was verified
    # separately by transcoding one recording to each and comparing transcripts.
    install_pipeline(monkeypatch, [])

    response = await client.post(
        "/assessments/parse", files={"file": (filename, WAV_BYTES, "application/octet-stream")}
    )

    assert response.status_code == 200
    assert response.json()["audioFilename"] == filename


async def test_the_rejection_says_what_would_have_worked(client):
    response = await client.post(
        "/assessments/parse", files={"file": ("notes.txt", b"hello", "text/plain")}
    )
    assert response.status_code == 415
    assert ".mp3" in response.json()["detail"]


async def test_a_non_audio_upload_is_refused(client):
    response = await client.post(
        "/assessments/parse", files={"file": ("notes.pdf", b"%PDF-1.4", "application/pdf")}
    )
    assert response.status_code == 415


async def test_a_filename_cannot_escape_the_temp_directory(client, monkeypatch):
    # The uploaded name is attacker-controlled; only its basename is ever used.
    install_pipeline(monkeypatch, [])

    response = await client.post(
        "/assessments/parse",
        files={"file": ("../../etc/passwd.wav", WAV_BYTES, "audio/wav")},
    )

    assert response.status_code == 200
    assert response.json()["audioFilename"] == "passwd.wav"


async def test_an_empty_upload_is_refused(client):
    response = await client.post(
        "/assessments/parse", files={"file": ("session.wav", b"", "audio/wav")}
    )
    assert response.status_code == 400


async def test_an_oversized_upload_is_refused(client, monkeypatch):
    monkeypatch.setattr(main.settings, "max_upload_mb", 1)

    response = await client.post(
        "/assessments/parse",
        files={"file": ("session.wav", b"\x00" * (2 * 1024 * 1024), "audio/wav")},
    )

    assert response.status_code == 413


# --------------------------------------------------------------------------- #
# The other three endpoints
# --------------------------------------------------------------------------- #
async def test_saving_returns_201_and_the_new_id(client, monkeypatch):
    install_db(monkeypatch)
    draft = StoredAssessment(transcript="t", assessment=make_assessment())

    response = await client.post("/assessments", json=draft.model_dump(mode="json"))

    assert response.status_code == 201
    assert response.json()["id"] == "507f1f77bcf86cd799439011"


async def test_saving_does_not_re_apply_the_confidence_gate(client, monkeypatch):
    # A clinician who corrected a misheard measurement must be able to save it.
    # The gate belongs where a machine produces values, not where a human does.
    install_db(monkeypatch)
    draft = StoredAssessment(
        assessment=make_assessment(),
        flags=ExtractionFlags.summarise([evidence("clinicalDetails.duration", 0.01)]),
    )

    response = await client.post("/assessments", json=draft.model_dump(mode="json"))

    assert response.status_code == 201


async def test_saving_rejects_an_unknown_field(client, monkeypatch):
    install_db(monkeypatch)
    draft = StoredAssessment(assessment=make_assessment()).model_dump(mode="json")
    draft["assessment"]["painScore"] = "7"

    response = await client.post("/assessments", json=draft)

    assert response.status_code == 422


async def test_a_missing_assessment_is_a_404(client, monkeypatch):
    install_db(monkeypatch)
    response = await client.get("/assessments/507f1f77bcf86cd799439011")
    assert response.status_code == 404


async def test_parse_is_not_swallowed_by_the_id_route(client, monkeypatch):
    # If /assessments/{id} were registered first, "parse" would be read as an id
    # and this would be a 405 or a 404 instead of a validation error.
    install_db(monkeypatch)
    response = await client.post("/assessments/parse")
    assert response.status_code == 422  # missing the file, not missing the route


async def test_an_unparseable_date_is_rejected(client, monkeypatch):
    install_db(monkeypatch)
    response = await client.get("/assessments", params={"date": "last tuesday"})
    assert response.status_code == 422


async def test_a_valid_date_is_passed_through_as_a_date(client, monkeypatch):
    seen = {}

    async def listing(day=None, *, limit=50, skip=0):
        seen["day"] = day
        seen["limit"] = limit
        return []

    install_db(monkeypatch, list_assessments=listing)

    response = await client.get("/assessments", params={"date": "2026-08-20", "limit": 5})

    assert response.status_code == 200
    assert str(seen["day"]) == "2026-08-20"
    assert seen["limit"] == 5


async def test_an_absurd_limit_is_rejected(client, monkeypatch):
    install_db(monkeypatch)
    response = await client.get("/assessments", params={"limit": 100_000})
    assert response.status_code == 422


# --------------------------------------------------------------------------- #
# Failure modes
# --------------------------------------------------------------------------- #
async def test_a_database_outage_is_503_not_500(client, monkeypatch):
    # The request was fine; the database was not. 500 would send the caller
    # looking for a bug in their payload.
    async def down(*args, **kwargs):
        raise ServerSelectionTimeoutError("no primary available")

    install_db(monkeypatch, save_assessment=down, get_assessment=down, list_assessments=down)

    saved = await client.post(
        "/assessments", json=StoredAssessment(assessment=make_assessment()).model_dump(mode="json")
    )
    fetched = await client.get("/assessments/507f1f77bcf86cd799439011")
    listed = await client.get("/assessments")

    assert [saved.status_code, fetched.status_code, listed.status_code] == [503, 503, 503]


async def test_health_reports_the_database_rather_than_always_passing(client, monkeypatch):
    monkeypatch.setattr(db, "ping", lambda: _false())
    degraded = await client.get("/health")
    assert degraded.status_code == 503
    assert degraded.json() == {"status": "degraded", "mongo": False}

    monkeypatch.setattr(db, "ping", lambda: _true())
    healthy = await client.get("/health")
    assert healthy.status_code == 200
    assert healthy.json() == {"status": "ok", "mongo": True}


async def _false() -> bool:
    return False


async def _true() -> bool:
    return True

# --------------------------------------------------------------------------- #
# The review UI
# --------------------------------------------------------------------------- #
async def test_the_ui_is_served_from_the_api_itself(client):
    # Same origin is what lets the page call the API with no CORS configuration
    # at all. If this ever moves to its own host, that stops being true.
    response = await client.get("/ui/")
    assert response.status_code == 200
    assert "<title>First Assessment</title>" in response.text


async def test_the_root_sends_you_to_the_ui(client):
    response = await client.get("/")
    assert response.status_code == 307
    assert response.headers["location"] == "/ui/"


async def test_the_ui_mount_does_not_shadow_the_api(client, monkeypatch):
    # Mounted last, on its own prefix. A mount registered too early swallows
    # everything under it.
    install_db(monkeypatch)
    assert (await client.get("/assessments")).status_code == 200

# --------------------------------------------------------------------------- #
# A section lost to a failed model call
# --------------------------------------------------------------------------- #
async def test_an_unavailable_section_is_a_422_even_when_the_rest_scored_well(
    client, monkeypatch
):
    """The failure mode that shipped a confident-looking, incomplete document.

    An empty section is an ordinary, correct answer when the clinician did not
    mention it. When the call that produces it failed instead, the document
    looks identical, and the confidence score goes *up*, because it averages
    only the fields that came back. So the status code cannot be left to the
    score to decide.
    """
    install_pipeline(
        monkeypatch,
        [evidence("objectiveAssessment.tests[0].left", 0.98)],
        failed=["clinicalDetails", "subjectiveAssessments"],
    )

    response = await client.post(
        "/assessments/parse", files={"file": ("session.wav", WAV_BYTES, "audio/wav")}
    )

    assert response.status_code == 422
    detail = response.json()["detail"]
    kinds = [d["type"] for d in detail]
    assert kinds.count("section_unavailable") == 2
    assert "low_confidence" not in kinds  # the one real field was fine
    assert [d["loc"] for d in detail] == [
        ["assessment", "clinicalDetails"],
        ["assessment", "subjectiveAssessments"],
    ]


async def test_the_unavailable_message_does_not_blame_the_recording(client, monkeypatch):
    install_pipeline(monkeypatch, [], failed=["patientAdvice"])

    response = await client.post(
        "/assessments/parse", files={"file": ("session.wav", WAV_BYTES, "audio/wav")}
    )

    error = response.json()["detail"][0]
    assert "not because the recording was silent" in error["msg"]
    assert error["ctx"]["section"] == "patientAdvice"


async def test_unavailable_sections_are_listed_before_low_confidence_fields(
    client, monkeypatch
):
    # A whole missing section outranks a single doubtful number.
    install_pipeline(
        monkeypatch,
        [evidence("objectiveAssessment.tests[0].left", 0.05)],
        failed=["patientAdvice"],
    )

    detail = (
        await client.post(
            "/assessments/parse", files={"file": ("session.wav", WAV_BYTES, "audio/wav")}
        )
    ).json()["detail"]

    assert [d["type"] for d in detail] == ["section_unavailable", "low_confidence"]


async def test_the_draft_still_comes_back_with_the_missing_section(client, monkeypatch):
    install_pipeline(monkeypatch, [], failed=["clinicalDetails"])

    body = (
        await client.post(
            "/assessments/parse", files={"file": ("session.wav", WAV_BYTES, "audio/wav")}
        )
    ).json()

    # Everything that did survive is still reviewable, and the flags say what
    # was lost so the UI can mark it rather than showing a blank section.
    assert tuple(body["assessment"].keys()) == SECTIONS
    assert body["flags"]["failedSections"] == ["clinicalDetails"]


async def test_a_total_extraction_failure_is_a_502_not_a_422(client, monkeypatch):
    """Nothing was wrong with the request; the provider did not answer.

    422 would send the caller looking for a problem in their audio. 502 says
    what is actually true: try again later.
    """
    install_pipeline(monkeypatch, [])

    def unavailable(*args, **kwargs):
        raise ExtractionUnavailable("every model call failed: 429 RESOURCE_EXHAUSTED")

    monkeypatch.setattr(main, "extract", unavailable)

    response = await client.post(
        "/assessments/parse", files={"file": ("session.wav", WAV_BYTES, "audio/wav")}
    )

    assert response.status_code == 502
    assert "429" in response.json()["detail"]


async def test_an_ordinary_extraction_failure_is_still_a_422(client, monkeypatch):
    # ExtractionUnavailable subclasses ExtractionFailed, so the handlers have to
    # be ordered correctly for this one to stay a 422.
    install_pipeline(monkeypatch, [])

    def refused(*args, **kwargs):
        raise ExtractionFailed("Empty transcript.")

    monkeypatch.setattr(main, "extract", refused)

    response = await client.post(
        "/assessments/parse", files={"file": ("session.wav", WAV_BYTES, "audio/wav")}
    )

    assert response.status_code == 422
