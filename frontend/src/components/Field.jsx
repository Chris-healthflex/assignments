import { severityOf } from "../lib/confidence";
import { useAssessment } from "../state/AssessmentContext";
import { ConfidencePill } from "./ConfidencePill";

/**
 * One editable contract value.
 *
 * Two shapes for the same thing: `row` for prose fields that need a label above
 * them, `cell` for a measurement sitting in a table where the column heading is
 * already the label. Both carry the same flag styling, so a misheard number
 * looks the same wherever the layout happens to put it.
 *
 * The explanation of *why* a field is flagged does not live here. It goes in the
 * inspector beside the transcript, because the useful version of that
 * explanation is the recording itself, and repeating it inside every table cell
 * would bury the measurements it is meant to draw attention to.
 */
export function Field({ path, label, multiline = false, placeholder = "-", variant = "row" }) {
  const { valueAt, update, flagFor, evidenceFor, confidenceAt, selected, select, readOnly } =
    useAssessment();

  const value = valueAt(path);
  const flag = flagFor(path);
  const evidence = evidenceFor(path);
  const severity = severityOf(flag);
  const isSelected = selected === path;
  const confidence = evidence || flag ? confidenceAt(path) : null;

  const shared = {
    id: path,
    value,
    placeholder,
    readOnly,
    "aria-invalid": Boolean(flag),
    "aria-label": label,
    onFocus: () => select(path),
    onClick: () => select(path),
    onChange: (event) => update(path, event.target.value),
  };

  const classes = [
    "field",
    `field-${variant}`,
    severity ? `flag-${severity}` : "",
    isSelected ? "is-selected" : "",
  ]
    .filter(Boolean)
    .join(" ");

  const control = multiline ? (
    <textarea {...shared} rows={Math.min(8, Math.max(2, Math.ceil(value.length / 68)))} />
  ) : (
    <input type="text" {...shared} />
  );

  if (variant === "cell") {
    return <div className={classes}>{control}</div>;
  }

  return (
    <div className={classes}>
      <label htmlFor={path}>
        {label}
        <ConfidencePill
          confidence={confidence}
          severity={severity}
          title={flag ? flag.msg : "Confidence in this value"}
        />
      </label>
      {control}
    </div>
  );
}
