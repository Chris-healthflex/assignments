"""Shared pytest fixtures. All tests run in stub mode (no Ollama / Mongo needed)."""
from __future__ import annotations

import os

os.environ.setdefault("USE_STUB_LLM", "1")
os.environ.setdefault("ALLOW_MEMORY_DB", "1")

import pytest


TRANSCRIPT = (
    "The patient presented with left knee pain, difficulty performing functional "
    "activities and difficulty walking along with ankle and back pain during prolonged "
    "walking following surgery. The patient was apparently normal eight months ago when "
    "she was involved in a road traffic accident resulting in a left tibial condol "
    "fracture and an avulsion ACL tear. Open reduction and internal fixation was "
    "performed by Dr. Hemant Kalyan, followed by four to six weeks of non-weight bearing "
    "and subsequent progressive loading. On assessment, a healed surgical scar was noted "
    "on the medial aspect of the knee, with restricted and painful knee flexion on over "
    "pressure, restricted extension and swelling. Patellar mobility was good, while hip "
    "range of motion was generally full and pain-free although left hip extension was "
    "restricted. Objective measurements showed left knee flexion of 124 degrees compared "
    "with 130 degrees on the right, left knee extension of 20 degrees compared with 5 "
    "degrees on the right, hip internal rotation of 45 degrees bilaterally, hip external "
    "rotation of 60 degrees bilaterally and ankle dorsiflexion of 4.5 degrees on the left "
    "compared with 12 degrees on the right. Physiotherapy was recommended once weekly for "
    "four sessions, with emphasis on restoring the extension, improving knee stability, "
    "strengthening the quadriceps, improving ankle mobility, and activating the posterior "
    "chain."
)


@pytest.fixture
def transcript() -> str:
    return TRANSCRIPT


@pytest.fixture
def wav_path() -> str:
    return os.path.join("data", "clinical_assessment.wav")
