import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

from app.models.assessment import FirstAssessment
from app.models.confidence import ExtractionResult

load_dotenv()


SYSTEM_PROMPT = """
You are a clinical documentation extraction agent.

Your task is to extract ONLY information explicitly supported by the supplied
clinical transcript and map it into the FirstAssessment schema.

You must also provide an internal confidence score for each top-level section.

STRICT RULES:
1. Never invent clinical facts, measurements, diagnoses, dates, scores, or treatment.
2. Never infer a value that is not supported by the transcript.
3. Preserve left/right/bilateral distinctions exactly.
4. Correct obvious speech-recognition errors only when the intended clinical
   terminology is unambiguous from the surrounding transcript.
5. Every string field must contain a string. If information is not available,
   use an empty string.
6. Every array field must always be an array.
7. Do not add fields that are not part of FirstAssessment.
8. Objective measurements must preserve the stated numerical values.
9. Do not convert or reinterpret measurements.
10. Do not fabricate target dates. If no target date is stated, use an empty string.
11. Clinical history should summarize the explicitly stated relevant history.
12. Chief complaint should contain the explicitly stated presenting complaints.
13. Subjective assessments should contain subjective symptoms/findings.
14. Objective tests should contain explicitly stated examination measurements
    and findings.
15. Goals must be separated correctly:
    - subjectiveGoals are patient-reported goals or desired functional outcomes.
    - objectiveGoals are clinician-stated measurable or treatment-oriented
      goals, such as restoring range of motion, improving stability,
      strengthening musculature, improving mobility, or activating the
      posterior chain.
    - Do not put clinician treatment objectives into subjectiveGoals.16. Recommendations should contain explicitly stated treatment recommendations.
17. Patient advice should contain only advice explicitly given to the patient.

CONFIDENCE RULES:
- Give each top-level section a confidence score from 0.0 to 1.0.
- Confidence measures the reliability of the EXTRACTION, not whether the
  section contains information.
- If a field or section is legitimately absent from the transcript and the
  correct output is an empty string or empty array, assign HIGH confidence
  (0.90-1.00) when the transcript provides enough context to determine that
  the information was not stated.
- Do NOT give low confidence merely because a field is empty.
- Use high confidence (0.90-1.00) when the information is explicitly stated
  and unambiguous, OR when its absence is clear from the transcript.
- Use medium confidence (0.70-0.89) when the information is supported but
  requires minor interpretation or normalization.
- Use low confidence (below 0.70) only when the extracted information is
  ambiguous, contradictory, corrupted beyond reliable interpretation, or
  otherwise genuinely uncertain.
- Do not increase confidence simply because information sounds clinically
  plausible.
- Do not decrease confidence simply because a valid field contains an empty
  string or an array is empty.

Return:
1. assessment: a FirstAssessment object
2. confidence: a ConfidenceReport object

The confidence object is INTERNAL validation metadata and must NOT be included
in the final API response.
"""


def get_extraction_model():
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not configured. Add it to the local .env file."
        )

    model_name = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")

    return ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key,
        temperature=0,
    )


def extract_assessment_with_confidence(
    transcript: str,
) -> tuple[FirstAssessment, object]:
    if not transcript.strip():
        raise ValueError("Transcript is empty.")

    model = get_extraction_model()

    structured_model = model.with_structured_output(ExtractionResult)

    response = structured_model.invoke(
        [
            ("system", SYSTEM_PROMPT),
            (
                "human",
                "Extract the clinical assessment and internal confidence "
                "scores from this transcript:\n\n"
                + transcript,
            ),
        ]
    )

    if not isinstance(response, ExtractionResult):
        raise RuntimeError(
            "Model did not return the expected ExtractionResult."
        )

    return response.assessment, response.confidence


def extract_assessment(transcript: str) -> FirstAssessment:
    assessment, _ = extract_assessment_with_confidence(transcript)
    return assessment