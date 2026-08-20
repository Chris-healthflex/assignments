/**
 * The confidence rule, mirrored from `FieldEvidence.confidence` on the server.
 *
 * The server does not send the combined score. It is a computed property there
 * on purpose, because storing it would freeze one version of the rule into
 * every saved record, which means the rule now exists in two places. The API sends
 * its own number for the fields it flagged, so the copy below is only used for
 * the fields that passed; the Node checks in `npm test`-less form (see
 * scripts/check-logic.mjs) pin it to the same behaviours the Python tests pin.
 */

/** Below this, a neighbouring word counts as destroyed rather than merely unclear. */
export const GARBLED = 0.25;

export function score(evidence) {
  if (!evidence || !evidence.evidenceFound) return 0;
  // A signal of exactly zero means "not reported", not "certain it is wrong".
  // Gemini often omits its own confidence, and treating that as 0 would zero
  // out every well-grounded field on the page.
  const reported = [evidence.modelConfidence, evidence.audioConfidence].filter(
    (signal) => signal != null && signal > 0,
  );
  let combined = reported.length ? Math.min(...reported) : 0;
  if (evidence.contextConfidence != null && evidence.contextConfidence < GARBLED) {
    combined = Math.min(combined, evidence.contextConfidence);
  }
  return combined;
}

/**
 * Three states, and only three get colour: a field is trusted, was heard badly,
 * or was never traced to the recording at all. The last is the serious one:
 * it means nothing in the transcript supports the value.
 */
export function severityOf(flag) {
  if (!flag) return null;
  // Both of these mean "nothing supports this", one at field level and one at
  // section level. A value the transcript does not back and a section nobody
  // could ask for are the same problem at different scales.
  return flag.type === "unverified_evidence" || flag.type === "section_unavailable"
    ? "bad"
    : "warn";
}

export const pct = (value) => `${Math.round(value * 100)}%`;
