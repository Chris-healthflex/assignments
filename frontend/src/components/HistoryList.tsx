import { useEffect, useState } from "react";
import { listAssessments } from "../api";
import type { SavedAssessment } from "../types";
import { AssessmentView } from "./AssessmentView";

export function HistoryList() {
  const [assessments, setAssessments] = useState<SavedAssessment[]>([]);
  const [selected, setSelected] = useState<SavedAssessment | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    listAssessments()
      .then(setAssessments)
      .catch(() => setError("Could not load saved assessments"))
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <p className="text-sm text-slate-500">Loading…</p>;
  if (error) return <p className="text-sm text-amber-700">{error}</p>;
  if (assessments.length === 0) {
    return <p className="text-sm text-slate-400">No saved assessments yet.</p>;
  }

  if (selected) {
    return (
      <div className="space-y-4">
        <button
          onClick={() => setSelected(null)}
          className="text-sm font-medium text-teal-700 hover:underline"
        >
          ← Back to list
        </button>
        <AssessmentView assessment={selected} />
      </div>
    );
  }

  return (
    <div className="space-y-2">
      {assessments.map((a) => (
        <button
          key={a.id}
          onClick={() => setSelected(a)}
          className="flex w-full items-center justify-between rounded-xl border border-slate-200 bg-white p-4 text-left shadow-sm transition hover:border-teal-300"
        >
          <div>
            <p className="text-sm font-medium text-slate-900">
              {a.clinicalDetails.chiefComplaint || "Untitled assessment"}
            </p>
            <p className="text-xs text-slate-500">{a.id}</p>
          </div>
          <span className="text-xs text-slate-400">
            {new Date(a.createdAt).toLocaleString()}
          </span>
        </button>
      ))}
    </div>
  );
}
