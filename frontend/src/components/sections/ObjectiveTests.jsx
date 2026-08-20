import { SectionCard } from "../SectionCard";
import { Table } from "../Table";
import { useAssessment } from "../../state/AssessmentContext";

/**
 * Left and right are adjacent on purpose. The clinical signal in a measurement
 * set like this is usually the difference between the two sides, and a layout
 * that separates them makes the reader hold numbers in their head to compare.
 *
 * `value` stays as its own column rather than being merged with left/right:
 * the contract has all three, and a measurement that is not sided (a single
 * total, a score) belongs in `value` with the sided columns left empty.
 */
const COLUMNS = [
  { key: "testName", label: "Test", width: "26%" },
  { key: "left", label: "Left", width: "9%" },
  { key: "right", label: "Right", width: "9%" },
  { key: "value", label: "Value", width: "9%" },
  { key: "unitName", label: "Unit", width: "12%" },
  { key: "comments", label: "Comments", multiline: true },
];

export function ObjectiveTests() {
  const { assessment } = useAssessment();
  const rows = assessment.objectiveAssessment?.tests ?? [];

  return (
    <SectionCard
      title="Objective assessment"
      section="objectiveAssessment"
      count={rows.length}
      empty="No measurements were taken in this recording."
    >
      <Table columns={COLUMNS} count={rows.length} basePath="objectiveAssessment.tests" />
    </SectionCard>
  );
}
