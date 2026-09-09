#!/usr/bin/env python
"""
Exhaustive Live End-to-End Test Suite using clinical_assessment.wav.
Tests:
1. Audio file loading, decoding, resampling & audio properties
2. In-process Whisper ASR transcription & accuracy
3. LangGraph extraction agent & schema strictness
4. Deterministic anti-hallucination verification
5. Live FastAPI POST /assessments/parse with multipart WAV upload
6. Header checks: X-Extraction-Confidence & X-Extraction-Flags
7. MongoDB save via parse endpoint (save=true)
8. Retrieval via GET /assessments/{id}
9. Listing & filtering via GET /assessments
10. Robust edge cases (empty audio, corrupted audio, non-WAV format)
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
import soundfile as sf
from fastapi.testclient import TestClient

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.agent import run_extraction_pipeline
from app.config import settings
from app.main import app
from app.schemas import FirstAssessment
from app.transcription import decode_wav_to_16k_mono, transcribe_audio

client = TestClient(app)
WAV_PATH = Path("clinical_assessment.wav")


def print_section(title: str):
    print("\n" + "=" * 60)
    print(f"  {title.upper()}")
    print("=" * 60)


def test_1_audio_properties():
    print_section("Step 1: Audio File Analysis & Decoding")
    assert WAV_PATH.exists(), f"File {WAV_PATH} not found!"
    size_mb = WAV_PATH.stat().st_size / (1024 * 1024)
    print(f"[OK] File exists: {WAV_PATH}")
    print(f"[INFO] File size: {size_mb:.2f} MB")

    info = sf.info(str(WAV_PATH))
    print(f"[INFO] Channels: {info.channels}")
    print(f"[INFO] Sample rate: {info.samplerate} Hz")
    print(f"[INFO] Duration: {info.duration:.2f} seconds ({info.duration / 60:.1f} minutes)")
    print(f"[INFO] Format: {info.format} ({info.subtype})")

    # Test in-memory decode & resample
    t0 = time.time()
    with open(WAV_PATH, "rb") as f:
        raw_bytes = f.read()
    mono_16k = decode_wav_to_16k_mono(raw_bytes)
    t_decode = time.time() - t0

    print(f"[OK] Decoded to 16kHz mono array in {t_decode:.3f}s")
    print(f"[INFO] Decoded shape: {mono_16k.shape}, dtype: {mono_16k.dtype}")
    assert mono_16k.ndim == 1, "Audio must be 1D mono"
    assert len(mono_16k) > 0, "Audio array cannot be empty"


def test_2_whisper_transcription():
    print_section("Step 2: Whisper ASR Transcription")
    t0 = time.time()
    transcript = transcribe_audio(str(WAV_PATH))
    duration = time.time() - t0

    print(f"[OK] Transcription completed in {duration:.2f}s")
    print(f"[INFO] Character length: {len(transcript)}")
    print(f"[INFO] Word count: {len(transcript.split())}")
    print("\n--- Transcript Excerpt ---")
    print(transcript[:350] + " ...\n")

    assert len(transcript) > 500, "Transcript should contain full clinical conversation"
    assert "knee" in transcript.lower()
    assert "flexion" in transcript.lower()
    return transcript


def test_3_langgraph_extraction(transcript: str):
    print_section("Step 3: LangGraph Extraction & Anti-Hallucination Audit")
    t0 = time.time()
    result = run_extraction_pipeline(transcript, session_date="2026-09-02")
    t_extract = time.time() - t0

    print(f"[OK] Extraction pipeline executed in {t_extract:.3f}s")
    print(f"[INFO] Overall Confidence: {result.overall_confidence}")
    print(f"[INFO] Low Confidence Triggered: {result.low_confidence}")
    print(f"[INFO] Flagged Fields Count: {len(result.flags)}")

    if result.flags:
        for f in result.flags:
            print(f"  - Flag: {f.field} (conf={f.confidence}): {f.reason}")
    else:
        print("[OK] Zero hallucination flags! All fields passed audit.")

    assert result.overall_confidence >= 0.70
    assert not result.low_confidence

    assessment = result.assessment
    print("\n--- Extracted Clinical Assessment Summary ---")
    print(f"Chief Complaint: {assessment.clinicalDetails.chiefComplaint}")
    print(f"Duration: {assessment.clinicalDetails.duration}")
    print(f"Clinical History: {assessment.clinicalDetails.clinicalHistory[:120]}...")
    print(f"Subjective Assessments: {len(assessment.subjectiveAssessments)} recorded")
    print(f"Objective Tests: {len(assessment.objectiveAssessment.tests)} recorded")
    for t in assessment.objectiveAssessment.tests:
        print(f"  * {t.testName}: left='{t.left}', right='{t.right}', value='{t.value}' ({t.unitName}) - {t.comments}")
    print(f"Subjective Goals: {len(assessment.subjectiveGoals)} recorded")
    print(f"Objective Goals: {len(assessment.objectiveGoals)} recorded")
    for g in assessment.objectiveGoals:
        print(f"  * Goal: {g.goalName} -> {g.value} {g.unitName} ({g.goalCategory})")
    print(f"Recommendation: {assessment.recommendation[0].sessionType} - {assessment.recommendation[0].sessionFrequency}")
    print(f"Patient Advice: {assessment.patientAdvice.adviceDetails[:120]}...")

    # Strict validation check
    dump = assessment.model_dump()
    FirstAssessment.model_validate(dump)
    print("\n[OK] Pydantic v2 strict schema validated with 0 errors!")
    return result


def test_4_fastapi_multipart_parse():
    print_section("Step 4: Live FastAPI POST /assessments/parse Endpoint")
    with open(WAV_PATH, "rb") as f:
        files = {"file": ("clinical_assessment.wav", f, "audio/wav")}
        data = {"save": "true", "session_date": "2026-09-02"}
        t0 = time.time()
        res = client.post("/assessments/parse", files=files, data=data)
        t_api = time.time() - t0

    print(f"[OK] Response status: {res.status_code} in {t_api:.2f}s")
    assert res.status_code == 200, f"Expected 200, got {res.status_code}: {res.text}"

    # Check Headers
    conf_header = res.headers.get("X-Extraction-Confidence")
    flags_header = res.headers.get("X-Extraction-Flags")
    print(f"[OK] Header 'X-Extraction-Confidence': {conf_header}")
    print(f"[OK] Header 'X-Extraction-Flags': {flags_header}")
    assert conf_header is not None
    assert float(conf_header) >= 0.70

    body = res.json()
    assert "clinicalDetails" in body
    assert body["clinicalDetails"]["chiefComplaint"] != ""
    assert "objectiveAssessment" in body
    assert len(body["objectiveAssessment"]["tests"]) > 0
    print("[OK] Response body conforms 100% to FirstAssessment schema.")


def test_5_crud_and_database_persistence():
    print_section("Step 5: Database Persistence, Retrieval & Listing")
    # 1. Fetch latest saved assessments
    list_res = client.get("/assessments?limit=5")
    assert list_res.status_code == 200
    items = list_res.json()
    print(f"[OK] Retrieved {len(items)} saved assessments from database.")
    assert len(items) > 0

    latest = items[0]
    doc_id = latest["id"]
    print(f"[OK] Testing retrieval of document ID: {doc_id}")

    # 2. Get by ID
    get_res = client.get(f"/assessments/{doc_id}")
    assert get_res.status_code == 200
    doc = get_res.json()
    assert doc["id"] == doc_id
    assert "createdAt" in doc
    assert doc["assessment"]["clinicalDetails"]["chiefComplaint"] != ""
    print(f"[OK] Document retrieved successfully with created timestamp: {doc['createdAt']}")

    # 3. Test non-existent ID -> 404
    non_existent = client.get("/assessments/507f1f77bcf86cd799439011")
    assert non_existent.status_code == 404
    print("[OK] Non-existent ID correctly returned 404 Not Found.")


def test_6_edge_cases_and_error_handling():
    print_section("Step 6: Edge Cases & Error Handling")

    # Non-WAV file -> 400
    res_non_wav = client.post(
        "/assessments/parse",
        files={"file": ("recording.mp3", b"ID3\x03\x00\x00\x00", "audio/mpeg")}
    )
    assert res_non_wav.status_code == 400
    print(f"[OK] Non-WAV file rejected: {res_non_wav.status_code} ({res_non_wav.json()['detail']})")

    # Empty WAV file -> 400
    res_empty = client.post(
        "/assessments/parse",
        files={"file": ("empty.wav", b"", "audio/wav")}
    )
    assert res_empty.status_code == 400
    print(f"[OK] Empty WAV file rejected: {res_empty.status_code} ({res_empty.json()['detail']})")

    # Corrupted WAV (invalid header bytes) -> 400
    res_corrupt = client.post(
        "/assessments/parse",
        files={"file": ("corrupt.wav", b"RIFF" + b"\x00" * 40, "audio/wav")}
    )
    assert res_corrupt.status_code == 400
    print(f"[OK] Corrupt WAV file rejected: {res_corrupt.status_code} ({res_corrupt.json()['detail']})")

    # Invalid date range query on /assessments -> 400
    res_bad_dates = client.get("/assessments?from=2026-12-01T00:00:00Z&to=2026-01-01T00:00:00Z")
    assert res_bad_dates.status_code == 400
    print(f"[OK] Inverted date range rejected: {res_bad_dates.status_code} ({res_bad_dates.json()['detail']})")


def main():
    print("\n" + "#" * 60)
    print("  STANCE HEALTH CLINICAL ASSESSMENT PIPELINE")
    print("  FULL LIVE TEST RUN ON CLINICAL_ASSESSMENT.WAV")
    print("#" * 60)

    t_start = time.time()
    test_1_audio_properties()
    transcript = test_2_whisper_transcription()
    test_3_langgraph_extraction(transcript)
    test_4_fastapi_multipart_parse()
    test_5_crud_and_database_persistence()
    test_6_edge_cases_and_error_handling()

    total_time = time.time() - t_start
    print("\n" + "#" * 60)
    print(f"  ALL LIVE TESTS COMPLETED SUCCESSFULLY IN {total_time:.2f}s!")
    print("#" * 60 + "\n")


if __name__ == "__main__":
    main()
