import { Link, useParams } from "react-router-dom";
import { useAssessment } from "../hooks/useAssessment";
import { AssessmentView } from "../components/AssessmentView";

export function AssessmentDetailPage() {
  const { id } = useParams<{ id: string }>();
  const { data, loading, error } = useAssessment(id);

  if (loading) {
    return (
      <div className="space-y-4">
        <div className="h-6 w-40 animate-pulse rounded bg-slate-100" />
        <div className="grid gap-4 sm:grid-cols-2">
          {[0, 1, 2, 3].map((i) => (
            <div key={i} className="h-32 animate-pulse rounded-xl border border-slate-200 bg-slate-100" />
          ))}
        </div>
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="space-y-4">
        <p className="text-sm text-amber-700">{error || "Assessment not found"}</p>
        <Link to="/history" className="text-sm font-medium text-teal-700 hover:underline">
          ← Back to list
        </Link>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <Link to="/history" className="text-sm font-medium text-teal-700 hover:underline">
        ← Back to list
      </Link>
      <AssessmentView assessment={data} />
    </div>
  );
}
