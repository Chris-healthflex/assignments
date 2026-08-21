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

export interface SavedAssessment extends FirstAssessment {
  id: string;
  createdAt: string;
}

export interface ParseErrorDetail {
  message: string;
  low_confidence_sections: string[];
}
