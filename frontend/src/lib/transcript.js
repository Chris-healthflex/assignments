/**
 * Locating a quoted span inside the transcript.
 *
 * The model quotes what it heard with its own punctuation; Whisper's transcript
 * punctuates independently. An exact search finds most of them, and the words
 * with anything between them finds the rest. A quote that matches neither is
 * reported as not found rather than approximated. The server has already made
 * the authoritative decision about whether the evidence exists, and guessing
 * harder here would only disagree with it on screen.
 */
const ESCAPE = /[.*+?^${}()|[\]\\]/g;

export function findSpan(text, quote) {
  if (!text || !quote) return null;

  const direct = text.toLowerCase().indexOf(quote.toLowerCase());
  if (direct >= 0) return [direct, direct + quote.length];

  const words = quote.toLowerCase().match(/[a-z0-9.]+/g);
  if (!words) return null;
  const loose = new RegExp(words.map((w) => w.replace(ESCAPE, "\\$&")).join("[^a-z0-9]+"), "i");
  const match = text.match(loose);
  return match ? [match.index, match.index + match[0].length] : null;
}

/** Split the transcript into the part before a span, the span, and the rest. */
export function splitAround(text, quote) {
  const span = findSpan(text, quote);
  if (!span) return null;
  return [text.slice(0, span[0]), text.slice(span[0], span[1]), text.slice(span[1])];
}
