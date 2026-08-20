import { Field } from "./Field";
import { useAssessment } from "../state/AssessmentContext";

/**
 * A repeated contract array, rendered as a table.
 *
 * Most of the contract's arrays are records with the same handful of keys, and
 * a table is what those are: the column heading carries the label once instead
 * of repeating it on every row, and, the part that matters clinically, left
 * and right sit in adjacent columns where an asymmetry is visible at a glance.
 *
 * Each section declares its own columns rather than this deriving them from the
 * data, so a column keeps its position and heading even when every value in it
 * is empty.
 */
export function Table({ columns, count, basePath }) {
  const { flagByPath } = useAssessment();

  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            {columns.map((column) => (
              <th key={column.key} style={column.width ? { width: column.width } : undefined}>
                {column.label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {Array.from({ length: count }, (_, index) => {
            const flaggedRow = columns.some((column) =>
              flagByPath.has(`${basePath}[${index}].${column.key}`),
            );
            return (
              <tr key={index} className={flaggedRow ? "row-flagged" : undefined}>
                {columns.map((column) => (
                  <td key={column.key} data-label={column.label}>
                    <Field
                      path={`${basePath}[${index}].${column.key}`}
                      label={column.label}
                      variant="cell"
                      multiline={column.multiline}
                      placeholder={column.placeholder ?? "-"}
                    />
                  </td>
                ))}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
