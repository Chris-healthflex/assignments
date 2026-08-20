import { createContext, useContext, useMemo } from "react";

import { score } from "../lib/confidence";
import { getAt, pathFromLoc, setAt } from "../lib/paths";

/**
 * Wiring between the contract and the form.
 *
 * Every input on the review screen is addressed by the same dotted path the
 * server uses for evidence, so a field knows, without anything being threaded
 * through props, whether it was flagged, what was quoted for it, and how sure
 * each signal was. The alternative was passing four props through six layers of
 * table markup.
 *
 * The assessment itself is owned by the screen above, not here: this provider
 * reads and writes it but does not hold it, so saving does not have to reach
 * into a context to find out what to send.
 */
const AssessmentContext = createContext(null);

export function AssessmentProvider({
  assessment,
  onChange,
  flags,
  detail = [],
  readOnly = false,
  selected,
  onSelect,
  children,
}) {
  const evidenceByPath = useMemo(
    () => new Map((flags?.fields ?? []).map((field) => [field.field, field])),
    [flags],
  );

  const flagByPath = useMemo(
    () =>
      new Map(
        detail.map((error) => [
          pathFromLoc(error.loc),
          { ...error.ctx, msg: error.msg, type: error.type },
        ]),
      ),
    [detail],
  );

  const value = useMemo(() => {
    const flagFor = (path) => flagByPath.get(path) ?? null;
    const evidenceFor = (path) => evidenceByPath.get(path) ?? null;

    return {
      assessment,
      readOnly,
      failedSections: flags?.failedSections ?? [],
      selected,
      select: onSelect,
      flagByPath,
      evidenceByPath,
      flagFor,
      evidenceFor,
      valueAt: (path) => getAt(assessment, path) ?? "",
      update: (path, next) => onChange(setAt(assessment, path, next)),
      /**
       * Flagged fields use the server's own number rather than the local rule.
       * Where the two could ever disagree, the server is the one that decided
       * the response status, so it is the one the screen should agree with.
       */
      confidenceAt: (path) => {
        const flag = flagFor(path);
        return flag ? flag.confidence : score(evidenceFor(path));
      },
    };
  }, [assessment, onChange, readOnly, selected, onSelect, flagByPath, evidenceByPath]);

  return <AssessmentContext.Provider value={value}>{children}</AssessmentContext.Provider>;
}

export function useAssessment() {
  const context = useContext(AssessmentContext);
  if (!context) throw new Error("useAssessment must be used inside an AssessmentProvider");
  return context;
}
