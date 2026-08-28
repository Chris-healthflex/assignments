"""
Builds sample_output/clinical_assessment_output.json.

IMPORTANT — how this differs from the real /assessments/parse path:
The `transcribe` step below is real (pocketsphinx, run on the actual uploaded WAV).
The `extract` step is NOT run through app/agent/graph.py's LLM call, because this
sandbox has no reachable LLM API (no ANTHROPIC_API_KEY / OPENAI_API_KEY, and
api.openai.com is network-blocked here). The JSON below was produced by manually
applying the exact same SYSTEM_PROMPT rules in app/agent/prompts.py to the real
transcript, then validated through the same ExtractionEnvelope/FirstAssessment
Pydantic models the LLM node would validate against. In a real deployment with an
LLM_PROVIDER key configured, run_pipeline() would produce this step automatically.
"""
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.transcription.whisper_service import transcribe
from app.schemas.first_assessment import ExtractionEnvelope

WAV_PATH = "sample_data/clinical_assessment.wav"

transcript_result = transcribe(WAV_PATH, engine="pocketsphinx")

assessment = {
    "clinicalDetails": {
        "clinicalHistory": (
            "Patient was involved in a road traffic accident approximately 8 months "
            "ago resulting in a left tibial condyle fracture, treated with open "
            "reduction and internal fixation (ORIF) with plates and screws, followed "
            "by 4-6 weeks of non-weight-bearing status and subsequent progressive "
            "loading. Despite 8 months since injury, the patient has not returned to "
            "full functional activity."
        ),
        "chiefComplaint": (
            "Left knee pain with difficulty performing functional activities and "
            "difficulty walking, along with back pain during prolonged walking."
        ),
        "duration": "8 months post-injury / post-operative",
    },
    "subjectiveAssessments": [
        {
            "testName": "Pain assessment",
            "conclusion": "Mild pain with mild irritability, particularly during prolonged walking and standing",
        }
    ],
    "objectiveAssessment": {
        "tests": [
            {
                "testName": "Inspection",
                "unitName": "",
                "value": "",
                "left": "",
                "right": "",
                "comments": "Surgical scar noted on the medial aspect of the knee, reported as tender (transcript unclear in this segment)",
            },
            {
                "testName": "Knee flexion / extension (qualitative)",
                "unitName": "",
                "value": "",
                "left": "Painful on overpressure; restricted extension; swelling noted",
                "right": "",
                "comments": "Transcript garbled in this segment (ASR errors); qualitative description only",
            },
            {
                "testName": "Hip range of motion",
                "unitName": "",
                "value": "",
                "left": "Extension restricted",
                "right": "Generally full/free",
                "comments": "Hip ROM described as generally full/free bilaterally except restricted left hip extension",
            },
            {
                "testName": "Knee flexion",
                "unitName": "degrees",
                "value": "",
                "left": "124",
                "right": "130",
                "comments": "",
            },
            {
                "testName": "Knee extension deficit",
                "unitName": "degrees",
                "value": "",
                "left": "20",
                "right": "5",
                "comments": "",
            },
            {
                "testName": "Hip rotation",
                "unitName": "degrees",
                "value": "",
                "left": "",
                "right": "",
                "comments": (
                    "Transcript too garbled to confidently attribute values to left/right "
                    "or to internal/external rotation. Raw ASR fragment: 'hip and ten and "
                    "rotation of forty five degrees ... eczema rotation of sixty degrees ... "
                    "affliction of full point five degrees on the left compared with twelve "
                    "degrees on the right'. Numbers mentioned (45, 60, ~X.5, 12) not extracted "
                    "as structured values to avoid hallucinating attribution."
                ),
            },
        ]
    },
    "subjectiveGoals": [
        {"goalDetails": "Restore knee extension", "targetDate": ""},
        {"goalDetails": "Improve knee stability and single-leg stability", "targetDate": ""},
        {"goalDetails": "Strengthen quadriceps and functional lower-limb musculature", "targetDate": ""},
        {"goalDetails": "Improve ankle mobility", "targetDate": ""},
        {"goalDetails": "Activate posterior chain musculature", "targetDate": ""},
    ],
    "objectiveGoals": [],
    "recommendation": [
        {"sessionType": "Physiotherapy", "sessionFrequency": "Once weekly"}
    ],
    "patientAdvice": {"adviceDetails": ""},
}

extraction_flags = [
    "objectiveAssessment.tests[0].comments",
    "objectiveAssessment.tests[1]",
    "objectiveAssessment.tests[5]",
    "recommendation[0].sessionFrequency",
    "transcript (low-confidence ASR engine used — pocketsphiynx fallback, not Whisper)",
]

envelope = ExtractionEnvelope(
    assessment=assessment,
    overall_confidence=0.35,
    extraction_flags=extraction_flags,
    transcript=transcript_result.text,
)

out = envelope.model_dump()
out["source_audio_filename"] = "clinical_assessment.wav"
out["transcription_engine"] = transcript_result.engine

os.makedirs("sample_output", exist_ok=True)
with open("sample_output/clinical_assessment_output.json", "w") as f:
    json.dump(out, f, indent=2)

print(json.dumps(out, indent=2))
