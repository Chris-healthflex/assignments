/**
 * The four endpoints, plus health.
 *
 * Base URL is empty because the page is served by the API itself; in `npm run
 * dev` Vite proxies the same paths through to the backend, so nothing here has
 * to know which of the two it is running under.
 */

export class ApiError extends Error {
  constructor(message, status) {
    super(message);
    this.status = status;
  }
}

async function readError(response) {
  try {
    const body = await response.json();
    if (typeof body.detail === "string") return body.detail;
    if (Array.isArray(body.detail)) return body.detail.map((d) => d.msg).join("; ");
  } catch {
    /* fall through to the status line */
  }
  return `${response.status} ${response.statusText}`;
}

/**
 * Upload a recording.
 *
 * A 422 here is not a failure to be thrown. It is the documented answer when
 * something was heard poorly, and it carries the draft along with the list of
 * fields to look at. Both outcomes are returned; only a genuine error throws.
 */
export async function parseRecording(file, { signal } = {}) {
  const body = new FormData();
  body.append("file", file, file.name);

  const response = await fetch("/assessments/parse", { method: "POST", body, signal });

  if (response.status === 200) {
    return { payload: await response.json(), detail: [] };
  }
  if (response.status === 422) {
    const payload = await response.json();
    // A 422 with no assessment is FastAPI rejecting the request itself (no file
    // attached, say), which is a real error rather than a low-confidence result.
    if (payload.assessment) return { payload, detail: payload.detail ?? [] };
    throw new ApiError(
      Array.isArray(payload.detail) ? payload.detail.map((d) => d.msg).join("; ") : "Invalid request",
      422,
    );
  }
  throw new ApiError(await readError(response), response.status);
}

export async function saveAssessment(draft) {
  const response = await fetch("/assessments", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(draft),
  });
  if (!response.ok) throw new ApiError(await readError(response), response.status);
  return response.json();
}

export async function listAssessments({ date, limit = 50 } = {}) {
  const query = new URLSearchParams({ limit: String(limit) });
  if (date) query.set("date", date);
  const response = await fetch(`/assessments?${query}`);
  if (!response.ok) throw new ApiError(await readError(response), response.status);
  return response.json();
}

export async function getAssessment(id) {
  const response = await fetch(`/assessments/${encodeURIComponent(id)}`);
  if (!response.ok) throw new ApiError(await readError(response), response.status);
  return response.json();
}

export async function checkHealth() {
  try {
    const response = await fetch("/health");
    const body = await response.json();
    return { reachable: true, mongo: Boolean(body.mongo) };
  } catch {
    return { reachable: false, mongo: false };
  }
}
