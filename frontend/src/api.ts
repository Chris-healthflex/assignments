import type { FirstAssessment, ParseDebugResult, ParseErrorDetail, SavedAssessment } from "./types";

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

export class ApiError extends Error {
  status: number;
  detail: unknown;

  constructor(status: number, detail: unknown) {
    super(typeof detail === "string" ? detail : "Request failed");
    this.status = status;
    this.detail = detail;
  }
}

async function handle<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new ApiError(response.status, body.detail);
  }
  return response.json() as Promise<T>;
}

export async function parseAssessment(file: File): Promise<ParseDebugResult> {
  const form = new FormData();
  form.append("file", file);

  const response = await fetch(`${API_BASE}/assessments/parse?include_debug=true`, {
    method: "POST",
    body: form,
  });
  return handle<ParseDebugResult>(response);
}

export async function saveAssessment(assessment: FirstAssessment): Promise<{ id: string }> {
  const response = await fetch(`${API_BASE}/assessments`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(assessment),
  });
  return handle<{ id: string }>(response);
}

export async function listAssessments(): Promise<SavedAssessment[]> {
  const response = await fetch(`${API_BASE}/assessments`);
  return handle<SavedAssessment[]>(response);
}

export async function getAssessment(id: string): Promise<SavedAssessment> {
  const response = await fetch(`${API_BASE}/assessments/${id}`);
  return handle<SavedAssessment>(response);
}

export function isParseErrorDetail(detail: unknown): detail is ParseErrorDetail {
  return (
    typeof detail === "object" &&
    detail !== null &&
    "low_confidence_sections" in detail
  );
}
