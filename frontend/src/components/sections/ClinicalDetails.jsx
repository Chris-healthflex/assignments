import { Field } from "../Field";
import { SectionCard } from "../SectionCard";

/**
 * The presenting picture: what brought the patient in, for how long, and the
 * background. Complaint and duration sit side by side because they are read
 * together -- "left knee pain" means something different at three days than at
 * eight months.
 */
export function ClinicalDetails() {
  return (
    <SectionCard title="Clinical details" section="clinicalDetails">
      <div className="pair">
        <Field path="clinicalDetails.chiefComplaint" label="Chief complaint" />
        <Field path="clinicalDetails.duration" label="Duration" />
      </div>
      <Field path="clinicalDetails.clinicalHistory" label="Clinical history" multiline />
    </SectionCard>
  );
}
