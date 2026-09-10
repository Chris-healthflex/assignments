import { FirstAssessment, AssessmentRecord, AssessmentListResponse, ParseResponseMeta } from './types/api';

const API_BASE = (import.meta as any).env?.VITE_API_URL || 'http://localhost:8000';

export async function parseAudio(file: File): Promise<{
  assessment: FirstAssessment;
  meta: ParseResponseMeta;
}> {
  const formData = new FormData();
  formData.append('file', file);

  const res = await fetch(`${API_BASE}/assessments/parse`, {
    method: 'POST',
    body: formData,
  });

  const confidenceHeader = res.headers.get('X-Extraction-Confidence');
  const modelHeader = res.headers.get('X-Extraction-Model') || 'default';
  const latencyHeader = res.headers.get('X-Pipeline-Latency-Ms') || '';

  const meta: ParseResponseMeta = {
    confidence: confidenceHeader ? parseFloat(confidenceHeader) : 0,
    model: modelHeader,
    latencyBreakdown: latencyHeader,
  };

  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    const message = errorData.detail || `Request failed with status ${res.status}`;
    const err: any = new Error(message);
    err.status = res.status;
    err.data = errorData;
    err.meta = meta;
    throw err;
  }

  const assessment = await res.json();
  return { assessment, meta };
}

export async function saveAssessment(assessment: FirstAssessment): Promise<AssessmentRecord> {
  const res = await fetch(`${API_BASE}/assessments`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(assessment),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `Failed to save assessment (${res.status})`);
  }

  return res.json();
}

export async function getAssessment(id: string): Promise<AssessmentRecord> {
  const res = await fetch(`${API_BASE}/assessments/${id}`);
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `Assessment not found (${res.status})`);
  }
  return res.json();
}

export async function listAssessments(
  from?: string,
  to?: string,
  limit: number = 20,
  skip: number = 0
): Promise<AssessmentListResponse> {
  const params = new URLSearchParams({
    limit: limit.toString(),
    skip: skip.toString(),
  });
  if (from) params.append('from', from);
  if (to) params.append('to', to);

  const res = await fetch(`${API_BASE}/assessments?${params.toString()}`);
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `Failed to load assessments (${res.status})`);
  }
  return res.json();
}
