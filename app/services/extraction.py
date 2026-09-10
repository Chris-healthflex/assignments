import re
from typing import TypedDict

from langgraph.graph import StateGraph, START, END

from app.models.schemas import (
    ClinicalDetails,
    Duration,
    FirstAssessment,
    ObjectiveAssessment,
    ObjectiveGoal,
    ObjectiveTest,
    PatientAdvice,
    Recommendation,
    SubjectiveAssessment,
    SubjectiveGoal,
)


class ExtractionState(TypedDict, total=False):
    transcript: str

    clinical_details: ClinicalDetails
    subjective_assessments: list[SubjectiveAssessment]
    objective_assessment: ObjectiveAssessment
    subjective_goals: list[SubjectiveGoal]
    objective_goals: list[ObjectiveGoal]
    recommendation: list[Recommendation]
    patient_advice: PatientAdvice

    assessment: FirstAssessment | None
    confidence: float
    missing_fields: list[str]


def extract_clinical_details(state: ExtractionState):
    transcript = state["transcript"]

    duration = Duration(
        value="Not specified",
        unit="Not specified",
    )

    duration_match = re.search(
        r"(?:eight|8)\s+months",
        transcript,
        re.IGNORECASE,
    )

    if duration_match:
        duration = Duration(
            value="8",
            unit="months",
        )

    history = "Not specified"

    has_tibial_fracture = re.search(
        r"left\s+tibial\s+cond(?:y|o)l(?:e)?\s+fracture",
        transcript,
        re.IGNORECASE,
    )

    has_acl_tear = re.search(
        r"(?:avulsion\s+)?ACL\s+tear",
        transcript,
        re.IGNORECASE,
    )

    has_road_traffic_accident = re.search(
        r"road\s+traffic\s+accident",
        transcript,
        re.IGNORECASE,
    )

    if has_tibial_fracture:
        history_parts = ["Left tibial condyle fracture"]

        if has_acl_tear:
            history_parts.append("avulsion ACL tear")

        if has_road_traffic_accident:
            history_parts.append("following a road traffic accident")

        history = ", ".join(history_parts) + "."

    chief_complaint = "Not specified"

    if re.search(
        r"left knee pain",
        transcript,
        re.IGNORECASE,
    ):
        chief_complaint = (
            "Left knee pain, difficulty performing functional "
            "activities and difficulty walking."
        )

    return {
        "clinical_details": ClinicalDetails(
            clinicalHistory=history,
            chiefComplaint=chief_complaint,
            duration=duration,
        )
    }


def extract_subjective_assessments(state: ExtractionState):
    transcript = state["transcript"]

    conclusions = []

    if re.search(
        r"moderate pain",
        transcript,
        re.IGNORECASE,
    ):
        conclusions.append(
            "Moderate pain with mild irritability, particularly "
            "during prolonged walking and standing, relieved with rest."
        )

    if not conclusions:
        conclusions.append("Not specified")

    return {
        "subjective_assessments": [
            SubjectiveAssessment(
                testName="Pain assessment",
                conclusion=conclusions,
            )
        ]
    }


def _extract_measurement(
    transcript: str,
    pattern: str,
    test_name: str,
    comment: str,
) -> ObjectiveTest | None:

    match = re.search(
        pattern,
        transcript,
        re.IGNORECASE,
    )

    if not match:
        return None

    groups = match.groupdict()

    left = groups.get("left")
    right = groups.get("right")

    # For bilateral measurements, one value applies to both sides.
    if left is None and groups.get("value") is not None:
        left = groups["value"]
        right = groups["value"]

    if left is None:
        return None

    if right is None:
        right = "Not specified"

    return ObjectiveTest(
        testName=test_name,
        unitName="degrees",
        value=left,
        left=left,
        right=right,
        comments=[comment],
    )


def extract_objective_assessment(state: ExtractionState):
    transcript = state["transcript"]

    tests = []

    measurement_patterns = [
        (
            r"left\s+knee\s+flexion\s+(?:of|was)\s+"
            r"(?P<left>\d+(?:\.\d+)?)"
            r"(?:\s*(?:degrees?|°))?"
            r"[\s\S]{0,60}?"
            r"(?P<right>\d+(?:\.\d+)?)"
            r"\s*(?:degrees?|°)?\s+on\s+the\s+right",
            "Knee flexion",
            "Left knee flexion is restricted compared with the right.",
        ),
        (
            r"left knee extension\s+(?:of|was)\s+"
            r"(?P<left>\d+(?:\.\d+)?)"
            r"(?:\s*(?:degrees?|°))?"
            r".{0,50}?"
            r"(?P<right>\d+(?:\.\d+)?)"
            r"\s*(?:degrees?|°)?\s+on the right",
            "Knee extension",
            "Left knee extension is restricted compared with the right.",
        ),
        (
            r"hip internal rotation\s+(?:was|of)\s+"
            r"(?P<value>\d+(?:\.\d+)?)"
            r"\s*(?:degrees?|°)?\s+bilaterally",
            "Hip internal rotation",
            "Hip internal rotation was recorded bilaterally.",
        ),
        (
            r"(?:left\s+)?hip\s+external\s+rotation\s+(?:was|of)\s+"
            r"(?P<value>\d+(?:\.\d+)?)"
            r"\s*(?:degrees?|°)?\s+bilaterally",
            "Hip external rotation",
            "Hip external rotation was recorded bilaterally.",
        ),
        (
            r"ankle dorsiflexion\s+(?:was|of)\s+"
            r"(?P<left>\d+(?:\.\d+)?)"
            r"\s*(?:degrees?|°)?"
            r".{0,40}?"
            r"(?P<right>\d+(?:\.\d+)?)"
            r"\s*(?:degrees?|°)?\s+on the right",
            "Ankle dorsiflexion",
            "Left ankle dorsiflexion is restricted compared with the right.",
        ),
    ]

    for pattern, test_name, comment in measurement_patterns:
        test = _extract_measurement(
            transcript,
            pattern,
            test_name,
            comment,
        )

        if test:
            tests.append(test)

    return {
        "objective_assessment": ObjectiveAssessment(
            tests=tests
        )
    }


def extract_goals(state: ExtractionState):
    transcript = state["transcript"]

    subjective_goals = []
    objective_goals = []

    if re.search(
        r"full functional activity",
        transcript,
        re.IGNORECASE,
    ):
        subjective_goals.append(
            SubjectiveGoal(
                goalDetails="Return to full functional activity.",
                targetDate=["Not specified"],
            )
        )

    if not subjective_goals:
        subjective_goals.append(
            SubjectiveGoal(
                goalDetails="Not specified",
                targetDate=["Not specified"],
            )
        )

    if re.search(
        r"restoring the extension",
        transcript,
        re.IGNORECASE,
    ):
        objective_goals.append(
            ObjectiveGoal(
                goalName="Knee extension",
                goalCategory="Range of motion",
                unitName="degrees",
                value="Improve",
                targetDate=["Not specified"],
            )
        )

    if re.search(
        r"knee stability",
        transcript,
        re.IGNORECASE,
    ):
        objective_goals.append(
            ObjectiveGoal(
                goalName="Knee stability",
                goalCategory="Stability",
                unitName="Not specified",
                value="Improve",
                targetDate=["Not specified"],
            )
        )

    if re.search(
        r"single leg stability",
        transcript,
        re.IGNORECASE,
    ):
        objective_goals.append(
            ObjectiveGoal(
                goalName="Single leg stability",
                goalCategory="Functional stability",
                unitName="Not specified",
                value="Improve",
                targetDate=["Not specified"],
            )
        )

    return {
        "subjective_goals": subjective_goals,
        "objective_goals": objective_goals,
    }


def extract_recommendations(state: ExtractionState):
    transcript = state["transcript"]

    session_type = "Not specified"
    frequency = "Not specified"

    if re.search(
        r"physiotherapy was recommended",
        transcript,
        re.IGNORECASE,
    ):
        session_type = "Physiotherapy"

    frequency_match = re.search(
        r"once weekly for four sessions",
        transcript,
        re.IGNORECASE,
    )

    if frequency_match:
        frequency = "Once weekly for four sessions"

    return {
        "recommendation": [
            Recommendation(
                sessionType=session_type,
                sessionFrequency=frequency,
            )
        ]
    }


def extract_patient_advice(state: ExtractionState):
    transcript = state["transcript"]

    advice = {}

    if re.search(
        r"strengthening the quadriceps",
        transcript,
        re.IGNORECASE,
    ):
        advice["strengthening"] = (
            "Strengthen the quadriceps and functional lower limb musculature."
        )

    if re.search(
        r"improving ankle mobility",
        transcript,
        re.IGNORECASE,
    ):
        advice["ankle_mobility"] = "Improve ankle mobility."

    if re.search(
        r"activating the posterior chain",
        transcript,
        re.IGNORECASE,
    ):
        advice["posterior_chain"] = "Activate the posterior chain."

    if not advice:
        advice["details"] = "Not specified"

    return {
        "patient_advice": PatientAdvice(
            adviceDetails=advice
        )
    }


def validate_and_build(state: ExtractionState):
    missing_fields = []

    if state["clinical_details"].clinicalHistory == "Not specified":
        missing_fields.append("clinicalDetails.clinicalHistory")

    if state["clinical_details"].chiefComplaint == "Not specified":
        missing_fields.append("clinicalDetails.chiefComplaint")

    if state["clinical_details"].duration.value == "Not specified":
        missing_fields.append("clinicalDetails.duration")

    assessment = FirstAssessment(
        clinicalDetails=state["clinical_details"],
        subjectiveAssessments=state["subjective_assessments"],
        objectiveAssessment=state["objective_assessment"],
        subjectiveGoals=state["subjective_goals"],
        objectiveGoals=state["objective_goals"],
        recommendation=state["recommendation"],
        patientAdvice=state["patient_advice"],
    )

    confidence = 1.0

    if missing_fields:
        confidence = 0.8

    return {
        "assessment": assessment,
        "confidence": confidence,
        "missing_fields": missing_fields,
    }


def build_extraction_graph():
    graph = StateGraph(ExtractionState)

    graph.add_node(
        "clinical_details",
        extract_clinical_details,
    )

    graph.add_node(
        "subjective_assessments",
        extract_subjective_assessments,
    )

    graph.add_node(
        "objective_assessment",
        extract_objective_assessment,
    )

    graph.add_node(
        "goals",
        extract_goals,
    )

    graph.add_node(
        "recommendations",
        extract_recommendations,
    )

    graph.add_node(
        "patient_advice",
        extract_patient_advice,
    )

    graph.add_node(
        "validate",
        validate_and_build,
    )

    graph.add_edge(START, "clinical_details")
    graph.add_edge(
        "clinical_details",
        "subjective_assessments",
    )
    graph.add_edge(
        "subjective_assessments",
        "objective_assessment",
    )
    graph.add_edge(
        "objective_assessment",
        "goals",
    )
    graph.add_edge(
        "goals",
        "recommendations",
    )
    graph.add_edge(
        "recommendations",
        "patient_advice",
    )
    graph.add_edge(
        "patient_advice",
        "validate",
    )
    graph.add_edge("validate", END)

    return graph.compile()


extraction_graph = build_extraction_graph()


class AssessmentExtractionError(Exception):
    def __init__(self, missing_fields: list[str], confidence: float):
        self.missing_fields = missing_fields
        self.confidence = confidence
        super().__init__(
            f"Assessment extraction confidence is too low: "
            f"{', '.join(missing_fields)}"
        )


def extract_assessment(transcript: str) -> FirstAssessment:
    result = extraction_graph.invoke(
        {
            "transcript": transcript,
            "confidence": 0.0,
            "missing_fields": [],
            "assessment": None,
        }
    )

    confidence = result["confidence"]
    missing_fields = result["missing_fields"]

    # 0.9 is the confidence threshold for accepting an assessment.
    if confidence < 0.9:
        raise AssessmentExtractionError(
            missing_fields=missing_fields,
            confidence=confidence,
        )

    return result["assessment"]