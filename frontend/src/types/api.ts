/**
 * Strict TypeScript contracts matching OpenAPI 3.1.0 schema generated from FastAPI.
 */

export interface ClinicalDetails {
  clinicalHistory: string;
  chiefComplaint: string;
  duration: string;
}

export interface SubjectiveAssessment {
  testName: string;
  conclusion: string;
}

export interface ObjectiveTest {
  testName: string;
  unitName: string;
  value: string;
  left: string;
  right: string;
  comments: string;
}

export interface ObjectiveAssessment {
  tests: ObjectiveTest[];
}

export interface SubjectiveGoal {
  goalDetails: string;
  targetDate: string;
}

export interface ObjectiveGoal {
  goalName: string;
  goalCategory: string;
  unitName: string;
  value: string;
  targetDate: string;
}

export interface Recommendation {
  sessionType: string;
  sessionFrequency: string;
}

export interface PatientAdvice {
  adviceDetails: string;
}

export interface FirstAssessment {
  clinicalDetails: ClinicalDetails;
  subjectiveAssessments: SubjectiveAssessment[];
  objectiveAssessment: ObjectiveAssessment;
  subjectiveGoals: SubjectiveGoal[];
  objectiveGoals: ObjectiveGoal[];
  recommendation: Recommendation[];
  patientAdvice: PatientAdvice;
}

export interface FieldEvidence {
  field: string;
  confidence: number;
  llm_confidence: number;
  grounded: boolean;
  evidence_span: string | null;
  reason: string;
}

export interface ConfidenceReport {
  overall: number;
  threshold: number;
  flags: FieldEvidence[];
  passed: boolean;
}

export interface AssessmentRecord {
  id: string;
  createdAt: string;
  assessment: FirstAssessment;
  confidence?: ConfidenceReport | null;
}

export interface AssessmentListResponse {
  items: AssessmentRecord[];
  total: number;
  limit: number;
  skip: number;
}

export interface ParseResponseMeta {
  confidence: number;
  model: string;
  latencyBreakdown: string;
}
