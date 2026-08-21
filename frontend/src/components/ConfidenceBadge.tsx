import { useState } from "react";
import { Card } from "./ui/Card";

export function ConfidenceBadge({
  confidence,
  flaggedCount,
}: {
  confidence: number;
  flaggedCount: number;
}) {
  const [showFormula, setShowFormula] = useState(false);
  const pct = Math.round(confidence * 100);
  const color =
    pct >= 85 ? "text-teal-700" : pct >= 60 ? "text-amber-700" : "text-red-700";
  const barColor =
    pct >= 85 ? "bg-teal-600" : pct >= 60 ? "bg-amber-500" : "bg-red-500";

  return (
    <Card className="p-4">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-xs font-medium uppercase tracking-wide text-slate-500">
            Extraction confidence
          </p>
          <p className={`text-2xl font-semibold ${color}`}>{pct}%</p>
        </div>
        <button
          onClick={() => setShowFormula((v) => !v)}
          className="text-xs font-medium text-slate-500 underline decoration-dotted hover:text-slate-700"
        >
          How is this calculated?
        </button>
      </div>

      <div className="mt-2 h-2 w-full overflow-hidden rounded-full bg-slate-100">
        <div className={`h-full ${barColor}`} style={{ width: `${pct}%` }} />
      </div>

      {flaggedCount > 0 && (
        <p className="mt-2 text-xs text-slate-500">
          {flaggedCount} of 7 sections had no supporting content in the recording.
        </p>
      )}

      {showFormula && (
        <p className="mt-2 border-t border-slate-100 pt-2 text-xs text-slate-500">
          (7 − sections with no supporting content in the transcript) ÷ 7. It's a
          coverage measure, not the model's own certainty — the model is
          instructed to leave a section blank and name it here rather than
          invent a value, so a lower score means the session genuinely didn't
          cover those sections, not that the model is unsure of what it did
          extract.
        </p>
      )}
    </Card>
  );
}
