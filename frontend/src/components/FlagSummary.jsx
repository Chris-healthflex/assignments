import { pct, severityOf } from "../lib/confidence";
import { humanise } from "../lib/format";
import { pathFromLoc } from "../lib/paths";
import { useAssessment } from "../state/AssessmentContext";

/**
 * Everything the service refused to vouch for, in one list.
 *
 * With ten flagged fields spread across seven sections, hunting for the amber
 * borders is the wrong job to give a clinician. This is the worklist: click an
 * entry and both the field and the words behind it come into view.
 *
 * Sorted by score, so the value the microphone destroyed is the first thing
 * read and a merely-quiet one is further down.
 */
export function FlagSummary({ detail }) {
  const { select, selected } = useAssessment();
  if (!detail.length) return null;

  const entries = detail
    .map((error) => ({
      path: pathFromLoc(error.loc),
      confidence: error.ctx?.confidence ?? 0,
      value: error.ctx?.section ? "whole section" : (error.ctx?.value ?? ""),
      severity: severityOf(error),
      msg: error.msg,
    }))
    .sort((a, b) => a.confidence - b.confidence);

  return (
    <section className="flags">
      <header className="panel-head">
        <h3>Needs review</h3>
        <span className="count">{entries.length}</span>
      </header>
      <ul>
        {entries.map((entry) => (
          <li key={entry.path}>
            <button
              type="button"
              className={`flag-entry ${entry.severity} ${selected === entry.path ? "is-selected" : ""}`}
              onClick={() => {
                select(entry.path);
                document.getElementById(entry.path)?.focus({ preventScroll: false });
              }}
              title={entry.msg}
            >
              <span className={`pill ${entry.severity}`}>{pct(entry.confidence)}</span>
              <span className="where">{describe(entry.path)}</span>
              <span className="what">{entry.value || "—"}</span>
            </button>
          </li>
        ))}
      </ul>
    </section>
  );
}

/** `objectiveAssessment.tests[1].left` -> `Objective assessment · test 2 · left`. */
function describe(path) {
  return path
    .split(".")
    .map((token) => {
      const match = token.match(/^(.+)\[(\d+)\]$/);
      if (!match) return humanise(token);
      const [, name, index] = match;
      return `${humanise(singular(name))} ${Number(index) + 1}`;
    })
    .join(" · ");
}

function singular(name) {
  if (name.endsWith("ies")) return `${name.slice(0, -3)}y`;
  return name.endsWith("s") ? name.slice(0, -1) : name;
}
