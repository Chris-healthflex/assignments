import { z } from "zod";

// Mirrors the FirstAssessment Pydantic schema exactly (app/schemas/first_assessment.py).
// Validating at the API boundary means a backend contract change surfaces as a
// clear parse error here, not a confusing crash deep inside a component.

export const sectionKeySchema = z.enum([
  "clinicalDetails",
  "subjectiveAssessments",
  "objectiveAssessment",
  "subjectiveGoals",
  "objectiveGoals",
  "recommendation",
  "patientAdvice",
]);

const clinicalDetailsSchema = z.object({
  clinicalHistory: z.string(),
  chiefComplaint: z.string(),
  duration: z.string(),
});

const subjectiveAssessmentSchema = z.object({
  testName: z.string(),
  conclusion: z.string(),
});

const objectiveTestSchema = z.object({
  testName: z.string(),
  unitName: z.string(),
  value: z.string(),
  left: z.string(),
  right: z.string(),
  comments: z.string(),
});

const objectiveAssessmentSchema = z.object({
  tests: z.array(objectiveTestSchema),
});

const subjectiveGoalSchema = z.object({
  goalDetails: z.string(),
  targetDate: z.string(),
});

const objectiveGoalSchema = z.object({
  goalName: z.string(),
  goalCategory: z.string(),
  unitName: z.string(),
  value: z.string(),
  targetDate: z.string(),
});

const recommendationSchema = z.object({
  sessionType: z.string(),
  sessionFrequency: z.string(),
});

const patientAdviceSchema = z.object({
  adviceDetails: z.string(),
});

export const firstAssessmentSchema = z.object({
  clinicalDetails: clinicalDetailsSchema,
  subjectiveAssessments: z.array(subjectiveAssessmentSchema),
  objectiveAssessment: objectiveAssessmentSchema,
  subjectiveGoals: z.array(subjectiveGoalSchema),
  objectiveGoals: z.array(objectiveGoalSchema),
  recommendation: z.array(recommendationSchema),
  patientAdvice: patientAdviceSchema,
});

export const savedAssessmentSchema = firstAssessmentSchema.extend({
  id: z.string(),
  createdAt: z.string(),
});

export const transcriptSegmentSchema = z.object({
  id: z.number(),
  start: z.number(),
  end: z.number(),
  text: z.string(),
});

export const fieldEvidenceSchema = z.object({
  field: z.string(),
  segmentIds: z.array(z.number()),
  quote: z.string(),
});

export const parseDebugResultSchema = z.object({
  assessment: firstAssessmentSchema,
  transcript: z.string(),
  segments: z.array(transcriptSegmentSchema),
  evidence: z.array(fieldEvidenceSchema),
  ungrounded_fields: z.array(z.string()),
  validation_issues: z.array(z.string()),
  attempts: z.number(),
  is_low_confidence: z.boolean(),
  low_confidence_sections: z.array(sectionKeySchema),
  confidence: z.number(),
});

export const createAssessmentResponseSchema = z.object({
  id: z.string(),
});
