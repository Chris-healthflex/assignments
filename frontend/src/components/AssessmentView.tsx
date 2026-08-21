import type {
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
        <h3 className="text-sm font-semibold uppercase tracking-wide text-teal-700">
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
  editable,
  multiline,
  onChange,
}: {
  label: string;
  value: string;
  editable?: boolean;
  multiline?: boolean;
  onChange?: (v: string) => void;
}) {
  if (editable && onChange) {
    const inputClass =
      "mt-1 w-full rounded-md border border-slate-300 px-2 py-1 text-sm text-slate-800 focus:border-teal-500 focus:outline-none";
    return (
      <label className="block">
        <span className="text-xs font-medium text-slate-500">{label}</span>
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
      <dt className="text-xs font-medium text-slate-500">{label}</dt>
      <dd className="text-sm text-slate-800">
        {value || <span className="text-slate-400">—</span>}
      </dd>
    </div>
  );
}

export function AssessmentView({
  assessment,
  flaggedSections = [],
  editable = false,
  onChange,
}: Props) {
  const isFlagged = (section: SectionKey) => flaggedSections.includes(section);

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

  return (
    <div className="grid gap-4 sm:grid-cols-2">
      <Section title="Clinical Details" flagged={isFlagged("clinicalDetails")}>
        <div className="space-y-2">
          <TextField
            label="Chief Complaint"
            value={assessment.clinicalDetails.chiefComplaint}
            editable={editable}
            onChange={(v) =>
              set("clinicalDetails", { ...assessment.clinicalDetails, chiefComplaint: v })
            }
          />
          <TextField
            label="Duration"
            value={assessment.clinicalDetails.duration}
            editable={editable}
            onChange={(v) =>
              set("clinicalDetails", { ...assessment.clinicalDetails, duration: v })
            }
          />
          <TextField
            label="Clinical History"
            value={assessment.clinicalDetails.clinicalHistory}
            editable={editable}
            multiline
            onChange={(v) =>
              set("clinicalDetails", { ...assessment.clinicalDetails, clinicalHistory: v })
            }
          />
        </div>
      </Section>

      <Section title="Patient Advice" flagged={isFlagged("patientAdvice")}>
        <TextField
          label="Advice"
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
            <div key={i} className="space-y-2 border-b border-slate-100 pb-2 last:border-0">
              <TextField
                label="Test"
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
            </div>
          ))}
          {assessment.subjectiveAssessments.length === 0 &&
            (editable ? (
              <button
                onClick={() =>
                  addBlank<SubjectiveAssessment>(
                    "subjectiveAssessments",
                    assessment.subjectiveAssessments,
                    { testName: "", conclusion: "" },
                  )
                }
                className="text-xs font-medium text-teal-700 hover:underline"
              >
                + Add manually
              </button>
            ) : (
              <p className="text-sm text-slate-400">None recorded</p>
            ))}
        </div>
      </Section>

      <Section title="Objective Assessment" flagged={isFlagged("objectiveAssessment")}>
        <div className="space-y-3">
          {assessment.objectiveAssessment.tests.map((test, i) => (
            <div key={i} className="grid grid-cols-2 gap-2 border-b border-slate-100 pb-2 last:border-0">
              <TextField
                label="Test"
                value={test.testName}
                editable={editable}
                onChange={(v) => updateObjectiveTest(i, { testName: v })}
              />
              <TextField
                label="Unit"
                value={test.unitName}
                editable={editable}
                onChange={(v) => updateObjectiveTest(i, { unitName: v })}
              />
              <TextField
                label="Left"
                value={test.left}
                editable={editable}
                onChange={(v) => updateObjectiveTest(i, { left: v })}
              />
              <TextField
                label="Right"
                value={test.right}
                editable={editable}
                onChange={(v) => updateObjectiveTest(i, { right: v })}
              />
            </div>
          ))}
          {assessment.objectiveAssessment.tests.length === 0 &&
            (editable ? (
              <button
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
                className="text-xs font-medium text-teal-700 hover:underline"
              >
                + Add manually
              </button>
            ) : (
              <p className="text-sm text-slate-400">No tests recorded</p>
            ))}
        </div>
      </Section>

      <Section title="Subjective Goals" flagged={isFlagged("subjectiveGoals")}>
        <div className="space-y-3">
          {assessment.subjectiveGoals.map((goal, i) => (
            <div key={i} className="space-y-2 border-b border-slate-100 pb-2 last:border-0">
              <TextField
                label="Goal"
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
            </div>
          ))}
          {assessment.subjectiveGoals.length === 0 &&
            (editable ? (
              <button
                onClick={() =>
                  addBlank<SubjectiveGoal>("subjectiveGoals", assessment.subjectiveGoals, {
                    goalDetails: "",
                    targetDate: "",
                  })
                }
                className="text-xs font-medium text-teal-700 hover:underline"
              >
                + Add manually
              </button>
            ) : (
              <p className="text-sm text-slate-400">None recorded</p>
            ))}
        </div>
      </Section>

      <Section title="Objective Goals" flagged={isFlagged("objectiveGoals")}>
        <div className="space-y-3">
          {assessment.objectiveGoals.map((goal, i) => (
            <div key={i} className="grid grid-cols-2 gap-2 border-b border-slate-100 pb-2 last:border-0">
              <TextField
                label="Goal"
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
            </div>
          ))}
          {assessment.objectiveGoals.length === 0 &&
            (editable ? (
              <button
                onClick={() =>
                  addBlank<ObjectiveGoal>("objectiveGoals", assessment.objectiveGoals, {
                    goalName: "",
                    goalCategory: "",
                    unitName: "",
                    value: "",
                    targetDate: "",
                  })
                }
                className="text-xs font-medium text-teal-700 hover:underline"
              >
                + Add manually
              </button>
            ) : (
              <p className="text-sm text-slate-400">None recorded</p>
            ))}
        </div>
      </Section>

      <Section title="Recommendations" flagged={isFlagged("recommendation")}>
        <div className="space-y-3">
          {assessment.recommendation.map((rec, i) => (
            <div key={i} className="grid grid-cols-2 gap-2 border-b border-slate-100 pb-2 last:border-0">
              <TextField
                label="Session Type"
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
            </div>
          ))}
          {assessment.recommendation.length === 0 &&
            (editable ? (
              <button
                onClick={() =>
                  addBlank<Recommendation>("recommendation", assessment.recommendation, {
                    sessionType: "",
                    sessionFrequency: "",
                  })
                }
                className="text-xs font-medium text-teal-700 hover:underline"
              >
                + Add manually
              </button>
            ) : (
              <p className="text-sm text-slate-400">None recorded</p>
            ))}
        </div>
      </Section>
    </div>
  );

  function updateObjectiveTest(index: number, patch: Partial<ObjectiveTest>) {
    const next = assessment.objectiveAssessment.tests.slice();
    next[index] = { ...next[index], ...patch };
    set("objectiveAssessment", { tests: next });
  }
}
