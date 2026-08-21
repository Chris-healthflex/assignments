import type { z } from "zod";
import type {
  firstAssessmentSchema,
  parseDebugResultSchema,
  savedAssessmentSchema,
  sectionKeySchema,
} from "./schemas";

export type SectionKey = z.infer<typeof sectionKeySchema>;
export type FirstAssessment = z.infer<typeof firstAssessmentSchema>;
export type SavedAssessment = z.infer<typeof savedAssessmentSchema>;
export type ParseDebugResult = z.infer<typeof parseDebugResultSchema>;

export type ClinicalDetails = FirstAssessment["clinicalDetails"];
export type SubjectiveAssessment = FirstAssessment["subjectiveAssessments"][number];
export type ObjectiveAssessment = FirstAssessment["objectiveAssessment"];
export type ObjectiveTest = ObjectiveAssessment["tests"][number];
export type SubjectiveGoal = FirstAssessment["subjectiveGoals"][number];
export type ObjectiveGoal = FirstAssessment["objectiveGoals"][number];
export type Recommendation = FirstAssessment["recommendation"][number];
export type PatientAdvice = FirstAssessment["patientAdvice"];
