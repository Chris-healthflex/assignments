import { SectionCard } from "../SectionCard";
import { Table } from "../Table";
import { useAssessment } from "../../state/AssessmentContext";

const COLUMNS = [
  { key: "testName", label: "Assessment", width: "32%" },
  { key: "conclusion", label: "Conclusion", multiline: true },
];

/** What the patient reported: named assessment, and what it concluded. */
export function SubjectiveAssessments() {
  const { assessment } = useAssessment();
  const rows = assessment.subjectiveAssessments ?? [];

  return (
    <SectionCard
      title="Subjective assessments"
      section="subjectiveAssessments"
      count={rows.length}
    >
      <Table columns={COLUMNS} count={rows.length} basePath="subjectiveAssessments" />
    </SectionCard>
  );
}
