import { useCallback, useEffect, useState } from "react";

import { listAssessments } from "../lib/api";
import { formatDateTime, todayIso } from "../lib/format";

/**
 * Saved assessments, by day.
 *
 * The date filters on when the assessment was captured. The contract holds no
 * date of its own -- its only dates are goal targets, which are intentions
 * rather than a record of when anything happened -- so the envelope's
 * `createdAt` is the only honest reading of "on this date".
 */
export function Browse({ onOpen, onError }) {
  const [date, setDate] = useState(todayIso());
  const [rows, setRows] = useState(null);
  const [loading, setLoading] = useState(false);

  const load = useCallback(
    async (day) => {
      setLoading(true);
      try {
        setRows(await listAssessments(day ? { date: day } : {}));
      } catch (error) {
        onError(`Could not load assessments — ${error.message}`);
        setRows([]);
      } finally {
        setLoading(false);
      }
    },
    [onError],
  );

  useEffect(() => {
    load(date);
    // Deliberately only on mount: after that, loading is driven by the buttons.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <div className="browse">
      <div className="controls">
        <input
          type="date"
          value={date}
          onChange={(event) => setDate(event.target.value)}
          aria-label="Filter by date (UTC)"
          title="Assessments are filed by the UTC day they were captured"
        />
        <span className="hint">UTC day</span>
        <button type="button" className="ghost" onClick={() => load(date)}>
          Load day
        </button>
        <button
          type="button"
          className="ghost"
          onClick={() => {
            setDate("");
            load(null);
          }}
        >
          Show latest
        </button>
      </div>

      {loading && <p className="empty">Loading…</p>}

      {!loading && rows?.length === 0 && (
        <p className="empty">
          {date ? `Nothing was recorded on ${date}.` : "No assessments have been saved yet."}
        </p>
      )}

      <ul className="results">
        {(rows ?? []).map((row) => {
          const flagged = (row.flags?.fields ?? []).length;
          const complaint = row.assessment?.clinicalDetails?.chiefComplaint;
          return (
            <li key={row.id}>
              <button type="button" onClick={() => onOpen(row.id)}>
                <span className="when">{formatDateTime(row.createdAt)}</span>
                <span className="what">
                  {complaint || row.audioFilename || "(no chief complaint recorded)"}
                </span>
                <span className="tag">{flagged} fields</span>
              </button>
            </li>
          );
        })}
      </ul>
    </div>
  );
}
