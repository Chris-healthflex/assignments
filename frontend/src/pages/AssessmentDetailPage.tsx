import { Link, useParams } from "react-router-dom";
import { useAssessment } from "../hooks/useAssessment";
import { AssessmentView } from "../components/AssessmentView";
import { useRegisterCommands } from "../components/CommandPalette";
import { Button } from "../components/ui/Button";

export function AssessmentDetailPage() {
  const { id } = useParams<{ id: string }>();
  const { data, loading, error } = useAssessment(id);

  function downloadJson() {
    if (!data) return;
    const blob = new Blob([JSON.stringify(data, null, 2)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `assessment-${data.id}.json`;
    anchor.click();
    URL.revokeObjectURL(url);
  }

  useRegisterCommands(
    data
      ? [
          {
            id: "export-saved-json",
            group: "Assessment",
            label: "Download this assessment as JSON",
            run: downloadJson,
          },
        ]
      : [],
    [data],
  );

  if (loading) {
    return (
      <div className="space-y-4">
        <div className="h-6 w-40 animate-pulse rounded bg-slate-100 dark:bg-slate-800" />
        <div className="grid gap-4 sm:grid-cols-2">
          {[0, 1, 2, 3].map((i) => (
            <div
              key={i}
              className="h-32 animate-pulse rounded-xl border border-slate-200 bg-slate-100 dark:border-slate-800 dark:bg-slate-800"
            />
          ))}
        </div>
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="space-y-4">
        <p className="text-sm text-amber-700 dark:text-amber-400">
          {error || "Assessment not found"}
        </p>
        <Link
          to="/history"
          className="text-sm font-medium text-teal-700 hover:underline dark:text-teal-400"
        >
          ← Back to list
        </Link>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between gap-4">
        <Link
          to="/history"
          className="text-sm font-medium text-teal-700 hover:underline dark:text-teal-400"
        >
          ← Back to list
        </Link>
        <div className="flex items-center gap-3">
          <span className="hidden font-mono text-xs text-slate-400 sm:inline">
            {data.id}
          </span>
          <Button variant="secondary" onClick={downloadJson}>
            Download JSON
          </Button>
        </div>
      </div>
      <AssessmentView assessment={data} />
    </div>
  );
}
