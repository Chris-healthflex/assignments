import { ClinicalDetails } from "./sections/ClinicalDetails";
import { ObjectiveGoals, SubjectiveGoals } from "./sections/Goals";
import { ObjectiveTests } from "./sections/ObjectiveTests";
import { PatientAdvice, Recommendations } from "./sections/PlanSections";
import { SubjectiveAssessments } from "./sections/SubjectiveAssessments";

/**
 * The seven contract sections, in contract order.
 *
 * Each one is laid out for what it actually holds rather than by a generic
 * walk of the JSON: measurements as a table with left and right adjacent, goals
 * with their targets, advice as prose. A generic renderer produced a wall of
 * identical text boxes in which a bilateral range-of-motion measurement looked
 * exactly like a free-text note, which is precisely the distinction a clinician
 * reading this needs to make quickly.
 */
export function AssessmentForm() {
  return (
    <div className="form">
      <ClinicalDetails />
      <SubjectiveAssessments />
      <ObjectiveTests />
      <SubjectiveGoals />
      <ObjectiveGoals />
      <Recommendations />
      <PatientAdvice />
    </div>
  );
}
