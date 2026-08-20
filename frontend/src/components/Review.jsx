import { useState } from "react";

import { pct } from "../lib/confidence";
import { formatDateTime } from "../lib/format";
import { AssessmentProvider } from "../state/AssessmentContext";
import { AssessmentForm } from "./AssessmentForm";
import { FlagSummary } from "./FlagSummary";
import { Inspector } from "./Inspector";

/**
 * The review screen.
 *
 * A 422 and a 200 render the same way, and that is deliberate: the status code
 * decides whether fields are marked, not whether the clinician gets to see the
 * draft. An error page with nothing on it to correct would be the wrong kind of
 * strict: the flagged fields are exactly what they need in front of them.
 */
export function Review({
  payload,
  detail,
  assessment,
  onChange,
  readOnly,
  onSave,
  saveState,
  onRestart,
}) {
  const [selected, setSelected] = useState(null);
  const flags = payload.flags ?? { fields: [], overallConfidence: 0 };

  const unsourced = detail.filter((error) => error.type === "unverified_evidence").length;
  const misheard = detail.length - unsourced;

  return (
    <AssessmentProvider
      assessment={assessment}
      onChange={onChange}
      flags={flags}
      detail={detail}
      readOnly={readOnly}
      selected={selected}
      onSelect={setSelected}
    >
      <div className="review-head">
        <div className="titling">
          <h2>{payload.audioFilename || "Assessment"}</h2>
          {payload.createdAt && <p className="when">{formatDateTime(payload.createdAt)}</p>}
        </div>

        <div className="tally">
          <span className="tag">{(flags.fields ?? []).length} fields extracted</span>
          <span className="tag">overall {pct(flags.overallConfidence ?? 0)}</span>
          {misheard > 0 && <span className="tag warn">{misheard} heard poorly</span>}
          {unsourced > 0 && <span className="tag bad">{unsourced} without a source</span>}
          {(flags.failedSections ?? []).length > 0 && (
            <span className="tag bad">{flags.failedSections.length} section(s) missing</span>
          )}
          {detail.length === 0 && (flags.failedSections ?? []).length === 0 && (
            <span className="tag ok">nothing flagged</span>
          )}
        </div>

        <div className="spacer" />

        {readOnly ? (
          <span className="tag">saved · read only</span>
        ) : (
          <>
            <button type="button" className="ghost" onClick={onRestart}>
              Start over
            </button>
            <button
              type="button"
              className="primary"
              onClick={onSave}
              disabled={saveState.status === "saving" || saveState.status === "saved"}
            >
              {saveState.status === "saving"
                ? "Saving..."
                : saveState.status === "saved"
                  ? `Saved · ${saveState.id.slice(-6)}`
                  : "Save assessment"}
            </button>
          </>
        )}
      </div>

      {(flags.failedSections ?? []).length > 0 && (
        <div className="incomplete">
          <b>This assessment is incomplete.</b>{" "}
          {flags.failedSections.join(", ")} could not be extracted. The model calls
          for those sections failed after retries. They are blank because the service
          could not ask, not because the recording was silent, and the confidence
          figure above covers only what did come back.
        </div>
      )}

      <div className="review-body">
        <div className="left">
          <FlagSummary detail={detail} />
          <AssessmentForm />
        </div>
        <Inspector transcript={payload.transcript ?? ""} />
      </div>
    </AssessmentProvider>
  );
}
