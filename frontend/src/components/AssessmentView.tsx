import type { FirstAssessment } from "../types";

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
      <h3 className="mb-3 text-sm font-semibold uppercase tracking-wide text-teal-700">
        {title}
      </h3>
      {children}
    </div>
  );
}

function Field({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <dt className="text-xs font-medium text-slate-500">{label}</dt>
      <dd className="text-sm text-slate-800">{value || <span className="text-slate-400">—</span>}</dd>
    </div>
  );
}

export function AssessmentView({ assessment }: { assessment: FirstAssessment }) {
  return (
    <div className="grid gap-4 sm:grid-cols-2">
      <Section title="Clinical Details">
        <dl className="space-y-2">
          <Field label="Chief Complaint" value={assessment.clinicalDetails.chiefComplaint} />
          <Field label="Duration" value={assessment.clinicalDetails.duration} />
          <Field label="Clinical History" value={assessment.clinicalDetails.clinicalHistory} />
        </dl>
      </Section>

      <Section title="Patient Advice">
        <p className="text-sm text-slate-800">
          {assessment.patientAdvice.adviceDetails || (
            <span className="text-slate-400">No advice recorded</span>
          )}
        </p>
      </Section>

      <Section title="Subjective Assessments">
        {assessment.subjectiveAssessments.length === 0 ? (
          <p className="text-sm text-slate-400">None recorded</p>
        ) : (
          <ul className="space-y-2">
            {assessment.subjectiveAssessments.map((item, i) => (
              <li key={i} className="text-sm">
                <span className="font-medium text-slate-900">{item.testName}</span>
                {" — "}
                <span className="text-slate-700">{item.conclusion}</span>
              </li>
            ))}
          </ul>
        )}
      </Section>

      <Section title="Objective Assessment">
        {assessment.objectiveAssessment.tests.length === 0 ? (
          <p className="text-sm text-slate-400">No tests recorded</p>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-sm">
              <thead>
                <tr className="text-xs text-slate-500">
                  <th className="pr-3 pb-1">Test</th>
                  <th className="pr-3 pb-1">Left</th>
                  <th className="pr-3 pb-1">Right</th>
                  <th className="pr-3 pb-1">Unit</th>
                </tr>
              </thead>
              <tbody>
                {assessment.objectiveAssessment.tests.map((test, i) => (
                  <tr key={i} className="border-t border-slate-100">
                    <td className="pr-3 py-1 font-medium text-slate-900">{test.testName}</td>
                    <td className="pr-3 py-1">{test.left || test.value || "—"}</td>
                    <td className="pr-3 py-1">{test.right || "—"}</td>
                    <td className="pr-3 py-1 text-slate-500">{test.unitName || "—"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Section>

      <Section title="Subjective Goals">
        {assessment.subjectiveGoals.length === 0 ? (
          <p className="text-sm text-slate-400">None recorded</p>
        ) : (
          <ul className="space-y-2">
            {assessment.subjectiveGoals.map((goal, i) => (
              <li key={i} className="text-sm">
                <span className="text-slate-800">{goal.goalDetails}</span>
                {goal.targetDate && (
                  <span className="ml-2 text-xs text-slate-500">by {goal.targetDate}</span>
                )}
              </li>
            ))}
          </ul>
        )}
      </Section>

      <Section title="Objective Goals">
        {assessment.objectiveGoals.length === 0 ? (
          <p className="text-sm text-slate-400">None recorded</p>
        ) : (
          <ul className="space-y-2">
            {assessment.objectiveGoals.map((goal, i) => (
              <li key={i} className="text-sm">
                <span className="font-medium text-slate-900">{goal.goalName}</span>
                {" · "}
                <span className="text-slate-600">{goal.goalCategory}</span>
                {" — "}
                <span>{goal.value} {goal.unitName}</span>
                {goal.targetDate && (
                  <span className="ml-2 text-xs text-slate-500">by {goal.targetDate}</span>
                )}
              </li>
            ))}
          </ul>
        )}
      </Section>

      <Section title="Recommendations">
        {assessment.recommendation.length === 0 ? (
          <p className="text-sm text-slate-400">None recorded</p>
        ) : (
          <ul className="space-y-2">
            {assessment.recommendation.map((rec, i) => (
              <li key={i} className="text-sm">
                <span className="font-medium text-slate-900">{rec.sessionType}</span>
                {" — "}
                <span className="text-slate-700">{rec.sessionFrequency}</span>
              </li>
            ))}
          </ul>
        )}
      </Section>
    </div>
  );
}
