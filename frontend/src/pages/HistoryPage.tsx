import { useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { useAssessments } from "../hooks/useAssessments";
import { useRegisterCommands } from "../components/CommandPalette";
import { Button } from "../components/ui/Button";
import { Card } from "../components/ui/Card";

const inputClass =
  "rounded-lg border border-slate-200 bg-white px-3 py-1.5 text-sm text-slate-700 outline-none transition focus:border-teal-400 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-200";

export function HistoryPage() {
  const [dateFrom, setDateFrom] = useState("");
  const [dateTo, setDateTo] = useState("");
  const [query, setQuery] = useState("");

  // Date bounds go to Mongo (the collection can grow unbounded); the free-text
  // filter stays client-side, since it only refines what's already on screen.
  const { data, loading, error, refetch } = useAssessments({ dateFrom, dateTo });

  const visible = useMemo(() => {
    const needle = query.trim().toLowerCase();
    if (!needle) return data;
    return data.filter((a) =>
      `${a.clinicalDetails.chiefComplaint} ${a.id}`.toLowerCase().includes(needle),
    );
  }, [data, query]);

  const hasFilters = Boolean(dateFrom || dateTo || query);

  function clearFilters() {
    setDateFrom("");
    setDateTo("");
    setQuery("");
  }

  useRegisterCommands(
    [
      {
        id: "refresh-history",
        group: "Saved Assessments",
        label: "Refresh list",
        run: () => void refetch(),
      },
      ...(hasFilters
        ? [
            {
              id: "clear-filters",
              group: "Saved Assessments",
              label: "Clear all filters",
              run: clearFilters,
            },
          ]
        : []),
    ],
    [refetch, hasFilters],
  );

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-end gap-3">
        <label className="flex flex-col gap-1">
          <span className="text-xs font-medium text-slate-500 dark:text-slate-400">
            Search
          </span>
          <input
            type="search"
            value={query}
            placeholder="Chief complaint or id"
            onChange={(e) => setQuery(e.target.value)}
            className={inputClass}
          />
        </label>

        <label className="flex flex-col gap-1">
          <span className="text-xs font-medium text-slate-500 dark:text-slate-400">
            From
          </span>
          <input
            type="date"
            value={dateFrom}
            max={dateTo || undefined}
            onChange={(e) => setDateFrom(e.target.value)}
            className={inputClass}
          />
        </label>

        <label className="flex flex-col gap-1">
          <span className="text-xs font-medium text-slate-500 dark:text-slate-400">
            To
          </span>
          <input
            type="date"
            value={dateTo}
            min={dateFrom || undefined}
            onChange={(e) => setDateTo(e.target.value)}
            className={inputClass}
          />
        </label>

        {hasFilters && (
          <Button variant="ghost" onClick={clearFilters}>
            Clear
          </Button>
        )}
      </div>

      {loading && <SkeletonList />}

      {error && (
        <div className="rounded-xl border border-amber-300 bg-amber-50 p-4 text-sm text-amber-900 dark:border-amber-800 dark:bg-amber-950/40 dark:text-amber-200">
          <div className="flex items-center justify-between gap-4">
            <p>{error}</p>
            <Button variant="ghost" onClick={() => void refetch()}>
              Retry
            </Button>
          </div>
        </div>
      )}

      {!loading && !error && visible.length === 0 && (
        <p className="text-sm text-slate-400">
          {hasFilters
            ? "No assessments match these filters."
            : "No saved assessments yet."}
        </p>
      )}

      {!loading && !error && visible.length > 0 && (
        <>
          <p className="text-xs text-slate-500 dark:text-slate-400">
            {visible.length} assessment{visible.length === 1 ? "" : "s"}
          </p>
          <div className="space-y-2">
            {visible.map((a) => (
              <Link key={a.id} to={`/history/${a.id}`} className="block">
                <Card className="flex items-center justify-between gap-4 p-4 transition hover:border-teal-300 dark:hover:border-teal-700">
                  <div className="min-w-0">
                    <p className="truncate text-sm font-medium text-slate-900 dark:text-slate-100">
                      {a.clinicalDetails.chiefComplaint || "Untitled assessment"}
                    </p>
                    <p className="truncate font-mono text-xs text-slate-500">{a.id}</p>
                  </div>
                  <span className="shrink-0 text-xs text-slate-400">
                    {new Date(a.createdAt).toLocaleString()}
                  </span>
                </Card>
              </Link>
            ))}
          </div>
        </>
      )}
    </div>
  );
}

function SkeletonList() {
  return (
    <div className="space-y-2">
      {[0, 1, 2].map((i) => (
        <div
          key={i}
          className="h-16 animate-pulse rounded-xl border border-slate-200 bg-slate-100 dark:border-slate-800 dark:bg-slate-800"
        />
      ))}
    </div>
  );
}
