import { Field } from "../Field";
import { SectionCard } from "../SectionCard";
import { Table } from "../Table";
import { useAssessment } from "../../state/AssessmentContext";

const RECOMMENDATION_COLUMNS = [
  { key: "sessionType", label: "Session type", width: "45%" },
  { key: "sessionFrequency", label: "Frequency" },
];

export function Recommendations() {
  const { assessment } = useAssessment();
  const rows = assessment.recommendation ?? [];

  return (
    <SectionCard
      title="Recommendation"
      section="recommendation"
      count={rows.length}
      empty="No treatment plan was stated in this recording."
    >
      <Table columns={RECOMMENDATION_COLUMNS} count={rows.length} basePath="recommendation" />
    </SectionCard>
  );
}

/**
 * Advice is a single free-text field in the contract, so it gets a single box
 * rather than being split into bullets the clinician never separated.
 */
export function PatientAdvice() {
  return (
    <SectionCard title="Patient advice" section="patientAdvice">
      <Field path="patientAdvice.adviceDetails" label="Advice given" multiline />
    </SectionCard>
  );
}
