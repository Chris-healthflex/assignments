import { renderToStaticMarkup } from "react-dom/server";

import { Review } from "../src/components/Review";

const noop = () => {};

/** Seven empty sections, exactly as the contract defines them. */
export function emptyAssessment() {
  return {
    clinicalDetails: { clinicalHistory: "", chiefComplaint: "", duration: "" },
    subjectiveAssessments: [],
    objectiveAssessment: { tests: [] },
    subjectiveGoals: [],
    objectiveGoals: [],
    recommendation: [],
    patientAdvice: { adviceDetails: "" },
  };
}

export function populatedAssessment() {
  const assessment = emptyAssessment();
  assessment.clinicalDetails = {
    clinicalHistory: "RTA eight months ago, left tibial condyle fracture.",
    chiefComplaint: "Left knee pain and stiffness",
    duration: "eight months",
  };
  assessment.subjectiveAssessments = [{ testName: "Pain", conclusion: "Worse on stairs" }];
  assessment.objectiveAssessment.tests = [
    {
      testName: "Knee flexion",
      unitName: "degrees",
      value: "",
      left: "124",
      right: "130",
      comments: "",
    },
  ];
  assessment.subjectiveGoals = [{ goalDetails: "Return to walking unaided", targetDate: "" }];
  assessment.objectiveGoals = [
    { goalName: "Knee flexion", goalCategory: "Range of motion", unitName: "degrees", value: "130", targetDate: "" },
  ];
  assessment.recommendation = [{ sessionType: "Physiotherapy", sessionFrequency: "once weekly" }];
  assessment.patientAdvice = { adviceDetails: "Ice after exercise." };
  return assessment;
}

export const TRANSCRIPT =
  "Objective measurements showed left knee flexion of 124 degrees with 130 degrees on the right.";

export function payloadFor(assessment) {
  return {
    id: "",
    createdAt: "2026-08-21T09:00:00Z",
    audioFilename: "clinical_assessment.wav",
    transcript: TRANSCRIPT,
    flags: {
      overallConfidence: 0.771,
      fields: [
        {
          field: "objectiveAssessment.tests[0].left",
          value: "124",
          evidence: "left knee flexion of 124 degrees",
          evidenceFound: true,
          modelConfidence: 0.95,
          audioConfidence: 0.5174,
          contextConfidence: 0.5174,
          reason: "",
        },
        {
          field: "objectiveAssessment.tests[0].unitName",
          value: "degrees",
          evidence: "",
          evidenceFound: false,
          modelConfidence: 0,
          audioConfidence: null,
          contextConfidence: null,
          reason: "No source quoted for this value.",
        },
      ],
      unresolvedFields: [],
      warnings: [],
    },
    assessment,
  };
}

/** What the API sends when one of the three model calls never returned. */
export const UNAVAILABLE_DETAIL = [
  {
    loc: ["assessment", "clinicalDetails"],
    msg:
      "This section could not be extracted: the model call for it failed after " +
      "retries. It is empty because we could not ask, not because the recording " +
      "was silent.",
    type: "section_unavailable",
    ctx: { section: "clinicalDetails" },
  },
];

export const DETAIL = [
  {
    loc: ["assessment", "objectiveAssessment", "tests", 0, "left"],
    msg: "Confidence 52% is below the 60% needed to return this value without review.",
    type: "low_confidence",
    ctx: {
      value: "124",
      evidence: "left knee flexion of 124 degrees",
      confidence: 0.5174,
      modelConfidence: 0.95,
      audioConfidence: 0.5174,
      contextConfidence: 0.5174,
    },
  },
  {
    loc: ["assessment", "objectiveAssessment", "tests", 0, "unitName"],
    msg: "No source quoted for this value.",
    type: "unverified_evidence",
    ctx: {
      value: "degrees",
      evidence: "",
      confidence: 0,
      modelConfidence: 0,
      audioConfidence: null,
      contextConfidence: null,
    },
  },
];

export function render({ assessment, payload, detail = [], readOnly = false }) {
  return renderToStaticMarkup(
    <Review
      payload={payload}
      detail={detail}
      assessment={assessment}
      onChange={noop}
      readOnly={readOnly}
      onSave={noop}
      saveState={{ status: "idle" }}
      onRestart={noop}
    />,
  );
}
