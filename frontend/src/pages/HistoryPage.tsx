import { Link } from "react-router-dom";
import { useAssessments } from "../hooks/useAssessments";
import { Card } from "../components/ui/Card";

export function HistoryPage() {
  const { data, loading, error } = useAssessments();

  if (loading) return <SkeletonList />;
  if (error) return <p className="text-sm text-amber-700">{error}</p>;
  if (data.length === 0) {
    return <p className="text-sm text-slate-400">No saved assessments yet.</p>;
  }

  return (
    <div className="space-y-2">
      {data.map((a) => (
        <Link key={a.id} to={`/history/${a.id}`}>
          <Card className="flex items-center justify-between p-4 transition hover:border-teal-300">
            <div>
              <p className="text-sm font-medium text-slate-900">
                {a.clinicalDetails.chiefComplaint || "Untitled assessment"}
              </p>
              <p className="text-xs text-slate-500">{a.id}</p>
            </div>
            <span className="text-xs text-slate-400">
              {new Date(a.createdAt).toLocaleString()}
            </span>
          </Card>
        </Link>
      ))}
    </div>
  );
}

function SkeletonList() {
  return (
    <div className="space-y-2">
      {[0, 1, 2].map((i) => (
        <div
          key={i}
          className="h-16 animate-pulse rounded-xl border border-slate-200 bg-slate-100"
        />
      ))}
    </div>
  );
}
