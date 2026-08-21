import { createContext, useContext } from "react";
import type {
  FieldEvidence,
  FirstAssessment,
  ObjectiveGoal,
  ObjectiveTest,
  Recommendation,
  SectionKey,
  SubjectiveAssessment,
  SubjectiveGoal,
} from "../types";
import { Badge } from "./ui/Badge";
import { Card } from "./ui/Card";

interface Props {
  assessment: FirstAssessment;
  flaggedSections?: SectionKey[];
  editable?: boolean;
  onChange?: (next: FirstAssessment) => void;
  /** Per-field citations from the extraction agent. */
  evidence?: FieldEvidence[];
  /** Field paths the agent filled but could not cite — possible hallucinations. */
  ungroundedFields?: string[];
  /** Called when the reviewer wants to see where a value came from. */
  onInspectField?: (path: string, label: string, segmentIds: number[]) => void;
  inspectedField?: string | null;
}

interface EvidenceContextValue {
  enabled: boolean;
  evidenceFor: (path: string) => FieldEvidence | null;
  isUngrounded: (path: string) => boolean;
  onInspect: (path: string, label: string, segmentIds: number[]) => void;
  inspectedField: string | null;
}

const EvidenceContext = createContext<EvidenceContextValue>({
  enabled: false,
  evidenceFor: () => null,
  isUngrounded: () => false,
  onInspect: () => {},
  inspectedField: null,
});

/** A citation on a parent path (e.g. `subjectiveAssessments[0]`) covers its children. */
function coversPath(evidenceField: string, path: string): boolean {
  if (evidenceField === path) return true;
  return (
    path.startsWith(evidenceField) &&
    (path[evidenceField.length] === "." || path[evidenceField.length] === "[")
  );
}

/**
 * Names the repeated row a field sits in.
 *
 * Six of the seven sections are lists, so a bare citation label reads "Test" or
 * "Left" five times over with nothing to tell them apart. Rows announce
 * themselves here instead of every field spelling out its own context.
 */
const FieldGroupContext = createContext<string | null>(null);

function FieldGroup({
  name,
  className,
  children,
}: {
  name: string;
  className?: string;
  children: React.ReactNode;
}) {
  return (
    <FieldGroupContext.Provider value={name.trim() || null}>
      <div className={className}>{children}</div>
    </FieldGroupContext.Provider>
  );
}

function Section({
  title,
  flagged,
  children,
}: {
  title: string;
  flagged?: boolean;
  children: React.ReactNode;
}) {
  return (
    <Card tone={flagged ? "flagged" : "default"} className="p-5">
      <div className="mb-3 flex items-center justify-between">
        <h3 className="text-sm font-semibold uppercase tracking-wide text-teal-700 dark:text-teal-400">
          {title}
        </h3>
        {flagged && <Badge tone="amber">Not stated in the recording</Badge>}
      </div>
      {children}
    </Card>
  );
}

function TextField({
  label,
  value,
  path,
  editable,
  multiline,
  onChange,
}: {
  label: string;
  value: string;
  path: string;
  editable?: boolean;
  multiline?: boolean;
  onChange?: (v: string) => void;
}) {
  const evidence = useContext(EvidenceContext);
  const group = useContext(FieldGroupContext);
  const hasValue = value.trim().length > 0;
  const citation = evidence.enabled && hasValue ? evidence.evidenceFor(path) : null;
  const ungrounded = evidence.enabled && hasValue && evidence.isUngrounded(path);
  const isInspected = evidence.inspectedField === path;
  // "Ankle dorsiflexion · Left" tells the reviewer which row they're verifying.
  const fullLabel = group ? `${group} · ${label}` : label;

  const marker = (
    <>
      {citation && (
        <button
          type="button"
          onClick={() => evidence.onInspect(path, fullLabel, citation.segmentIds)}
          title={citation.quote || "Show the transcript this came from"}
          aria-label={`Show transcript evidence for ${fullLabel}`}
          className={`ml-1.5 rounded px-1 py-px align-middle font-mono text-[10px] transition ${
            isInspected
              ? "bg-teal-600 text-white"
              : "bg-teal-100 text-teal-700 hover:bg-teal-200 dark:bg-teal-900/60 dark:text-teal-300 dark:hover:bg-teal-900"
          }`}
        >
          ⏱ {citation.segmentIds.join(", ") || "—"}
        </button>
      )}
      {ungrounded && (
        <span
          title="The model filled this in but could not point to anywhere in the recording. Verify before saving."
          aria-label={`${fullLabel} has no transcript evidence`}
          className="ml-1.5 rounded bg-amber-100 px-1 py-px align-middle text-[10px] font-medium text-amber-800 dark:bg-amber-900/60 dark:text-amber-300"
        >
          ⚠ unverified
        </span>
      )}
    </>
  );

  if (editable && onChange) {
    const inputClass = `mt-1 w-full rounded-md border px-2 py-1 text-sm focus:outline-none ${
      ungrounded
        ? "border-amber-400 bg-amber-50/50 text-slate-800 focus:border-amber-500 dark:bg-amber-950/30 dark:text-slate-100"
        : "border-slate-300 text-slate-800 focus:border-teal-500 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-100"
    }`;

    return (
      <label className="block">
        <span className="text-xs font-medium text-slate-500 dark:text-slate-400">
          {label}
        </span>
        {marker}
        {multiline ? (
          <textarea
            className={inputClass}
            rows={2}
            value={value}
            onChange={(e) => onChange(e.target.value)}
          />
        ) : (
          <input
            className={inputClass}
            value={value}
            onChange={(e) => onChange(e.target.value)}
          />
        )}
      </label>
    );
  }

  return (
    <div>
      <dt className="text-xs font-medium text-slate-500 dark:text-slate-400">
        {label}
        {marker}
      </dt>
      <dd className="text-sm text-slate-800 dark:text-slate-200">
        {value || <span className="text-slate-400">—</span>}
      </dd>
    </div>
  );
}

function AddButton({ onClick }: { onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      className="text-xs font-medium text-teal-700 hover:underline dark:text-teal-400"
    >
      + Add manually
    </button>
  );
}

function EmptyNote({ children = "None recorded" }: { children?: string }) {
  return <p className="text-sm text-slate-400 dark:text-slate-500">{children}</p>;
}

export function AssessmentView({
  assessment,
  flaggedSections = [],
  editable = false,
  onChange,
  evidence = [],
  ungroundedFields = [],
  onInspectField,
  inspectedField = null,
}: Props) {
  const isFlagged = (section: SectionKey) => flaggedSections.includes(section);

  const ungroundedSet = new Set(ungroundedFields);
  const evidenceContext: EvidenceContextValue = {
    enabled: evidence.length > 0 || ungroundedFields.length > 0,
    evidenceFor: (path) =>
      evidence.find((entry) => coversPath(entry.field, path)) ?? null,
    isUngrounded: (path) => ungroundedSet.has(path),
    onInspect: onInspectField ?? (() => {}),
    inspectedField,
  };

  function set<K extends keyof FirstAssessment>(key: K, value: FirstAssessment[K]) {
    onChange?.({ ...assessment, [key]: value });
  }

  function updateListItem<T>(
    key: keyof FirstAssessment,
    list: T[],
    index: number,
    patch: Partial<T>,
  ) {
    const next = list.slice();
    next[index] = { ...next[index], ...patch };
    set(key, next as FirstAssessment[typeof key]);
  }

  function addBlank<T>(key: keyof FirstAssessment, list: T[], blank: T) {
    set(key, [...list, blank] as FirstAssessment[typeof key]);
  }

  function updateObjectiveTest(index: number, patch: Partial<ObjectiveTest>) {
    const next = assessment.objectiveAssessment.tests.slice();
    next[index] = { ...next[index], ...patch };
    set("objectiveAssessment", { tests: next });
  }

  return (
    <EvidenceContext.Provider value={evidenceContext}>
      <div className="grid gap-4 sm:grid-cols-2">
        <Section title="Clinical Details" flagged={isFlagged("clinicalDetails")}>
          <div className="space-y-2">
            <TextField
              label="Chief Complaint"
              path="clinicalDetails.chiefComplaint"
              value={assessment.clinicalDetails.chiefComplaint}
              editable={editable}
              onChange={(v) =>
                set("clinicalDetails", {
                  ...assessment.clinicalDetails,
                  chiefComplaint: v,
                })
              }
            />
            <TextField
              label="Duration"
              path="clinicalDetails.duration"
              value={assessment.clinicalDetails.duration}
              editable={editable}
              onChange={(v) =>
                set("clinicalDetails", { ...assessment.clinicalDetails, duration: v })
              }
            />
            <TextField
              label="Clinical History"
              path="clinicalDetails.clinicalHistory"
              value={assessment.clinicalDetails.clinicalHistory}
              editable={editable}
              multiline
              onChange={(v) =>
                set("clinicalDetails", {
                  ...assessment.clinicalDetails,
                  clinicalHistory: v,
                })
              }
            />
          </div>
        </Section>

        <Section title="Patient Advice" flagged={isFlagged("patientAdvice")}>
          <TextField
            label="Advice"
            path="patientAdvice.adviceDetails"
            value={assessment.patientAdvice.adviceDetails}
            editable={editable}
            multiline
            onChange={(v) => set("patientAdvice", { adviceDetails: v })}
          />
        </Section>

        <Section
          title="Subjective Assessments"
          flagged={isFlagged("subjectiveAssessments")}
        >
          <div className="space-y-3">
            {assessment.subjectiveAssessments.map((item, i) => (
              <FieldGroup
                key={i}
                name={item.testName || `Assessment ${i + 1}`}
                className="space-y-2 border-b border-slate-100 pb-2 last:border-0 dark:border-slate-800"
              >
                <TextField
                  label="Test"
                  path={`subjectiveAssessments[${i}].testName`}
                  value={item.testName}
                  editable={editable}
                  onChange={(v) =>
                    updateListItem<SubjectiveAssessment>(
                      "subjectiveAssessments",
                      assessment.subjectiveAssessments,
                      i,
                      { testName: v },
                    )
                  }
                />
                <TextField
                  label="Conclusion"
                  path={`subjectiveAssessments[${i}].conclusion`}
                  value={item.conclusion}
                  editable={editable}
                  onChange={(v) =>
                    updateListItem<SubjectiveAssessment>(
                      "subjectiveAssessments",
                      assessment.subjectiveAssessments,
                      i,
                      { conclusion: v },
                    )
                  }
                />
              </FieldGroup>
            ))}
            {assessment.subjectiveAssessments.length === 0 &&
              (editable ? (
                <AddButton
                  onClick={() =>
                    addBlank<SubjectiveAssessment>(
                      "subjectiveAssessments",
                      assessment.subjectiveAssessments,
                      { testName: "", conclusion: "" },
                    )
                  }
                />
              ) : (
                <EmptyNote />
              ))}
          </div>
        </Section>

        <Section title="Objective Assessment" flagged={isFlagged("objectiveAssessment")}>
          <div className="space-y-3">
            {assessment.objectiveAssessment.tests.map((test, i) => (
              <FieldGroup
                key={i}
                name={test.testName || `Test ${i + 1}`}
                className="grid grid-cols-2 gap-2 border-b border-slate-100 pb-2 last:border-0 dark:border-slate-800"
              >
                <TextField
                  label="Test"
                  path={`objectiveAssessment.tests[${i}].testName`}
                  value={test.testName}
                  editable={editable}
                  onChange={(v) => updateObjectiveTest(i, { testName: v })}
                />
                <TextField
                  label="Unit"
                  path={`objectiveAssessment.tests[${i}].unitName`}
                  value={test.unitName}
                  editable={editable}
                  onChange={(v) => updateObjectiveTest(i, { unitName: v })}
                />
                <TextField
                  label="Left"
                  path={`objectiveAssessment.tests[${i}].left`}
                  value={test.left}
                  editable={editable}
                  onChange={(v) => updateObjectiveTest(i, { left: v })}
                />
                <TextField
                  label="Right"
                  path={`objectiveAssessment.tests[${i}].right`}
                  value={test.right}
                  editable={editable}
                  onChange={(v) => updateObjectiveTest(i, { right: v })}
                />
              </FieldGroup>
            ))}
            {assessment.objectiveAssessment.tests.length === 0 &&
              (editable ? (
                <AddButton
                  onClick={() =>
                    set("objectiveAssessment", {
                      tests: [
                        {
                          testName: "",
                          unitName: "",
                          value: "",
                          left: "",
                          right: "",
                          comments: "",
                        },
                      ],
                    })
                  }
                />
              ) : (
                <EmptyNote>No tests recorded</EmptyNote>
              ))}
          </div>
        </Section>

        <Section title="Subjective Goals" flagged={isFlagged("subjectiveGoals")}>
          <div className="space-y-3">
            {assessment.subjectiveGoals.map((goal, i) => (
              <FieldGroup
                key={i}
                name={goal.goalDetails || `Goal ${i + 1}`}
                className="space-y-2 border-b border-slate-100 pb-2 last:border-0 dark:border-slate-800"
              >
                <TextField
                  label="Goal"
                  path={`subjectiveGoals[${i}].goalDetails`}
                  value={goal.goalDetails}
                  editable={editable}
                  onChange={(v) =>
                    updateListItem<SubjectiveGoal>(
                      "subjectiveGoals",
                      assessment.subjectiveGoals,
                      i,
                      { goalDetails: v },
                    )
                  }
                />
                <TextField
                  label="Target Date"
                  path={`subjectiveGoals[${i}].targetDate`}
                  value={goal.targetDate}
                  editable={editable}
                  onChange={(v) =>
                    updateListItem<SubjectiveGoal>(
                      "subjectiveGoals",
                      assessment.subjectiveGoals,
                      i,
                      { targetDate: v },
                    )
                  }
                />
              </FieldGroup>
            ))}
            {assessment.subjectiveGoals.length === 0 &&
              (editable ? (
                <AddButton
                  onClick={() =>
                    addBlank<SubjectiveGoal>(
                      "subjectiveGoals",
                      assessment.subjectiveGoals,
                      { goalDetails: "", targetDate: "" },
                    )
                  }
                />
              ) : (
                <EmptyNote />
              ))}
          </div>
        </Section>

        <Section title="Objective Goals" flagged={isFlagged("objectiveGoals")}>
          <div className="space-y-3">
            {assessment.objectiveGoals.map((goal, i) => (
              <FieldGroup
                key={i}
                name={goal.goalName || `Goal ${i + 1}`}
                className="grid grid-cols-2 gap-2 border-b border-slate-100 pb-2 last:border-0 dark:border-slate-800"
              >
                <TextField
                  label="Goal"
                  path={`objectiveGoals[${i}].goalName`}
                  value={goal.goalName}
                  editable={editable}
                  onChange={(v) =>
                    updateListItem<ObjectiveGoal>(
                      "objectiveGoals",
                      assessment.objectiveGoals,
                      i,
                      { goalName: v },
                    )
                  }
                />
                <TextField
                  label="Category"
                  path={`objectiveGoals[${i}].goalCategory`}
                  value={goal.goalCategory}
                  editable={editable}
                  onChange={(v) =>
                    updateListItem<ObjectiveGoal>(
                      "objectiveGoals",
                      assessment.objectiveGoals,
                      i,
                      { goalCategory: v },
                    )
                  }
                />
                <TextField
                  label="Value"
                  path={`objectiveGoals[${i}].value`}
                  value={goal.value}
                  editable={editable}
                  onChange={(v) =>
                    updateListItem<ObjectiveGoal>(
                      "objectiveGoals",
                      assessment.objectiveGoals,
                      i,
                      { value: v },
                    )
                  }
                />
                <TextField
                  label="Target Date"
                  path={`objectiveGoals[${i}].targetDate`}
                  value={goal.targetDate}
                  editable={editable}
                  onChange={(v) =>
                    updateListItem<ObjectiveGoal>(
                      "objectiveGoals",
                      assessment.objectiveGoals,
                      i,
                      { targetDate: v },
                    )
                  }
                />
              </FieldGroup>
            ))}
            {assessment.objectiveGoals.length === 0 &&
              (editable ? (
                <AddButton
                  onClick={() =>
                    addBlank<ObjectiveGoal>(
                      "objectiveGoals",
                      assessment.objectiveGoals,
                      {
                        goalName: "",
                        goalCategory: "",
                        unitName: "",
                        value: "",
                        targetDate: "",
                      },
                    )
                  }
                />
              ) : (
                <EmptyNote />
              ))}
          </div>
        </Section>

        <Section title="Recommendations" flagged={isFlagged("recommendation")}>
          <div className="space-y-3">
            {assessment.recommendation.map((rec, i) => (
              <FieldGroup
                key={i}
                name={rec.sessionType || `Recommendation ${i + 1}`}
                className="grid grid-cols-2 gap-2 border-b border-slate-100 pb-2 last:border-0 dark:border-slate-800"
              >
                <TextField
                  label="Session Type"
                  path={`recommendation[${i}].sessionType`}
                  value={rec.sessionType}
                  editable={editable}
                  onChange={(v) =>
                    updateListItem<Recommendation>(
                      "recommendation",
                      assessment.recommendation,
                      i,
                      { sessionType: v },
                    )
                  }
                />
                <TextField
                  label="Frequency"
                  path={`recommendation[${i}].sessionFrequency`}
                  value={rec.sessionFrequency}
                  editable={editable}
                  onChange={(v) =>
                    updateListItem<Recommendation>(
                      "recommendation",
                      assessment.recommendation,
                      i,
                      { sessionFrequency: v },
                    )
                  }
                />
              </FieldGroup>
            ))}
            {assessment.recommendation.length === 0 &&
              (editable ? (
                <AddButton
                  onClick={() =>
                    addBlank<Recommendation>(
                      "recommendation",
                      assessment.recommendation,
                      { sessionType: "", sessionFrequency: "" },
                    )
                  }
                />
              ) : (
                <EmptyNote />
              ))}
          </div>
        </Section>
      </div>
    </EvidenceContext.Provider>
  );
}
