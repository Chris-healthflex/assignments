import { z } from "zod";
import {
  createAssessmentResponseSchema,
  parseDebugResultSchema,
  savedAssessmentSchema,
} from "./schemas";
import type { FirstAssessment, ParseDebugResult, SavedAssessment } from "./types";

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

export class ApiShapeError extends Error {
  constructor(context: string, cause: z.ZodError) {
    super(`Unexpected response shape from ${context}: ${cause.message}`);
  }
}

async function handle<T>(response: Response, schema: z.ZodType<T>, context: string): Promise<T> {
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new ApiError(response.status, body.detail);
  }
  const json = await response.json();
  const parsed = schema.safeParse(json);
  if (!parsed.success) {
    throw new ApiShapeError(context, parsed.error);
  }
  return parsed.data;
}

export async function parseAssessment(file: File): Promise<ParseDebugResult> {
  const form = new FormData();
  form.append("file", file);

  const response = await fetch(`${API_BASE}/assessments/parse?include_debug=true`, {
    method: "POST",
    body: form,
  });
  return handle(response, parseDebugResultSchema, "POST /assessments/parse");
}

export async function saveAssessment(assessment: FirstAssessment): Promise<{ id: string }> {
  const response = await fetch(`${API_BASE}/assessments`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(assessment),
  });
  return handle(response, createAssessmentResponseSchema, "POST /assessments");
}

export async function listAssessments(): Promise<SavedAssessment[]> {
  const response = await fetch(`${API_BASE}/assessments`);
  return handle(response, z.array(savedAssessmentSchema), "GET /assessments");
}

export async function getAssessment(id: string): Promise<SavedAssessment> {
  const response = await fetch(`${API_BASE}/assessments/${id}`);
  return handle(response, savedAssessmentSchema, `GET /assessments/${id}`);
}
