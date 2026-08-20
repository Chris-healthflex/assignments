import { useAssessment } from "../state/AssessmentContext";

/**
 * One of the seven contract sections.
 *
 * There are two ways a section can be empty and they mean opposite things. The
 * ordinary one is that the clinician did not mention it, which the brief treats
 * as the correct answer rather than a gap. The other is that the model call
 * producing this section never returned -- and on screen those look identical
 * unless something says otherwise, which is what the second branch below is.
 */
export function SectionCard({
  title,
  section,
  count,
  children,
  empty = "Nothing stated in the recording.",
}) {
  const { failedSections } = useAssessment();
  const unavailable = section != null && failedSections.includes(section);
  const isEmpty = count === 0;

  return (
    <section
      className={`section${unavailable ? " section-unavailable" : ""}`}
      aria-labelledby={`section-${title}`}
    >
      <header className="section-head">
        <h3 id={`section-${title}`}>{title}</h3>
        {count != null && !unavailable && <span className="count">{count}</span>}
        {unavailable && <span className="tag bad">not extracted</span>}
      </header>

      {unavailable ? (
        <p className="unavailable">
          This section could not be extracted — the model call for it failed after
          retries. It is blank because we could not ask, <b>not</b> because the
          recording was silent about it. Re-run the recording to fill it in.
        </p>
      ) : isEmpty ? (
        <p className="empty">{empty}</p>
      ) : (
        children
      )}
    </section>
  );
}
