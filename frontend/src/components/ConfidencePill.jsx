import { pct } from "../lib/confidence";

/**
 * The score for one field.
 *
 * Rendered only where there is evidence to score. A field the recording never
 * mentioned has no confidence to report, and showing "0%" beside it would read
 * as "we are certain this is wrong" rather than "nobody said it".
 */
export function ConfidencePill({ confidence, severity, title }) {
  if (confidence == null) return null;
  return (
    <span className={`pill ${severity ?? ""}`} title={title}>
      {pct(confidence)}
    </span>
  );
}
