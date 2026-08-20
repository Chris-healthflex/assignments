/** Small presentation helpers. Nothing here decides anything. */

/** `chiefComplaint` -> `Chief complaint`. */
export function humanise(key) {
  const spaced = key.replace(/([A-Z])/g, " $1").toLowerCase().trim();
  return spaced.charAt(0).toUpperCase() + spaced.slice(1);
}

export function formatDateTime(iso) {
  if (!iso) return "";
  const when = new Date(iso);
  return Number.isNaN(when.getTime()) ? iso : when.toLocaleString();
}

/**
 * Today, as a UTC calendar day.
 *
 * Deliberately UTC and not local. `GET /assessments?date=` filters on a UTC day
 * because that is what the stored `createdAt` is, so a picker defaulting to the
 * local date disagrees with the service for part of every day -- at 00:44 in
 * UTC+5:30 the local date is already tomorrow while everything saved in the last
 * five hours is filed under today. Matching the server is the honest default;
 * the control is labelled UTC so the difference is visible rather than
 * surprising.
 */
export function todayIso() {
  return new Date().toISOString().slice(0, 10);
}

export function elapsed(seconds) {
  const minutes = Math.floor(seconds / 60);
  return `${minutes}:${String(seconds % 60).padStart(2, "0")}`;
}
