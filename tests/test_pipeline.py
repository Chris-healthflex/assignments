import pytest
from app.services.extraction import ground_node, normalize_node, audit_node, AgentState
from app.models.assessment import FirstAssessment


def test_pipeline_nodes_deterministic_flow():
    sample_transcript = (
        "The patient presented with left knee pain following an accident eight months ago. "
        "Active left knee flexion was 124 degrees and right was 130 degrees. "
        "Left knee extension was 20 degrees. "
        "Physiotherapy was recommended once weekly for four sessions."
    )

    draft = {
        "clinicalDetails": {
            "chiefComplaint": "Left knee pain",
            "clinicalHistory": "Road traffic accident eight months ago",
            "duration": "8 months",
        },
        "subjectiveAssessments": [],
        "objectiveAssessment": {
            "tests": [
                {
                    "testName": "Knee Flexion",
                    "unitName": "degrees",
                    "value": "",
                    "left": "124",
                    "right": "130",
                    "comments": "",
                },
                {
                    "testName": "Knee Extension",
                    "unitName": "degrees",
                    "value": "",
                    "left": "20",
                    "right": "-5",  # '-5' might not match 'negic 5' directly or test ungrounded
                    "comments": "",
                },
            ]
        },
        "subjectiveGoals": [],
        "objectiveGoals": [],
        "recommendation": [
            {
                "sessionType": "Physiotherapy",
                "sessionFrequency": "once weekly for four sessions",
            }
        ],
        "patientAdvice": {
            "adviceDetails": "",
        },
    }

    raw_scores = [
        {"field": "clinicalDetails.chiefComplaint", "llm_confidence": 0.95, "reason": "Explicit"},
        {"field": "clinicalDetails.duration", "llm_confidence": 0.90, "reason": "Explicit"},
        {"field": "objectiveAssessment.tests[0].left", "llm_confidence": 0.95, "reason": "Explicit"},
        {"field": "objectiveAssessment.tests[0].right", "llm_confidence": 0.95, "reason": "Explicit"},
        {"field": "objectiveAssessment.tests[1].left", "llm_confidence": 0.95, "reason": "Explicit"},
        {"field": "recommendation[0].sessionFrequency", "llm_confidence": 0.90, "reason": "Explicit"},
    ]

    state: AgentState = {
        "transcript": sample_transcript,
        "draft": draft,
        "raw_scores": raw_scores,
    }

    # 1. Ground node
    grounded_state = ground_node(state)
    assert "field_evidence" in grounded_state
    evidence_list = grounded_state["field_evidence"]
    assert len(evidence_list) > 0

    # Verify 124 left knee flexion was grounded
    flex_left = next((e for e in evidence_list if e.field == "objectiveAssessment.tests[0].left"), None)
    assert flex_left is not None
    assert flex_left.grounded is True
    assert flex_left.confidence == 0.95
    assert flex_left.evidence_span is not None

    # 2. Normalize node
    norm_state = normalize_node(grounded_state)
    assert "assessment" in norm_state
    assert isinstance(norm_state["assessment"], FirstAssessment)
    assert norm_state["assessment"].clinicalDetails.chiefComplaint == "Left knee pain"

    # 3. Audit node
    audit_state = audit_node(norm_state)
    assert "confidence_report" in audit_state
    report = audit_state["confidence_report"]
    assert isinstance(report.overall, float)
    assert isinstance(report.flags, list)
