import { SectionCard } from "../SectionCard";
import { Table } from "../Table";
import { useAssessment } from "../../state/AssessmentContext";

/**
 * Target dates get a plain text column rather than a date picker.
 *
 * The contract types them as strings, and the recording rarely states one.
 * "In about three months" is a real answer that no date input will hold. A
 * picker would also invite someone to fill in a date the clinician never gave,
 * which is the specific thing this system exists to prevent.
 */
const SUBJECTIVE_COLUMNS = [
  { key: "goalDetails", label: "Goal", multiline: true },
  { key: "targetDate", label: "Target date", width: "22%", placeholder: "not stated" },
];

const OBJECTIVE_COLUMNS = [
  { key: "goalName", label: "Goal", width: "28%" },
  { key: "goalCategory", label: "Category", width: "18%" },
  { key: "value", label: "Value", width: "12%" },
  { key: "unitName", label: "Unit", width: "14%" },
  { key: "targetDate", label: "Target date", width: "18%", placeholder: "not stated" },
];

export function SubjectiveGoals() {
  const { assessment } = useAssessment();
  const rows = assessment.subjectiveGoals ?? [];

  return (
    <SectionCard
      title="Subjective goals"
      section="subjectiveGoals"
      count={rows.length}
      empty="No goals in the patient's own terms were recorded."
    >
      <Table columns={SUBJECTIVE_COLUMNS} count={rows.length} basePath="subjectiveGoals" />
    </SectionCard>
  );
}

export function ObjectiveGoals() {
  const { assessment } = useAssessment();
  const rows = assessment.objectiveGoals ?? [];

  return (
    <SectionCard
      title="Objective goals"
      section="objectiveGoals"
      count={rows.length}
      empty="No measurable targets were set in this recording."
    >
      <Table columns={OBJECTIVE_COLUMNS} count={rows.length} basePath="objectiveGoals" />
    </SectionCard>
  );
}
