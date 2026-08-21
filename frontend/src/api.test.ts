import { afterEach, describe, expect, it, vi } from "vitest";
import { ApiError, ApiShapeError, getAssessment, listAssessments, parseAssessment } from "./api";

function jsonResponse(body: unknown, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

const validAssessment = {
  clinicalDetails: { clinicalHistory: "", chiefComplaint: "Knee pain", duration: "" },
  subjectiveAssessments: [],
  objectiveAssessment: { tests: [] },
  subjectiveGoals: [],
  objectiveGoals: [],
  recommendation: [],
  patientAdvice: { adviceDetails: "" },
};

afterEach(() => {
  vi.restoreAllMocks();
});

describe("parseAssessment", () => {
  it("returns the parsed result when the response matches the schema", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        jsonResponse({
          assessment: validAssessment,
          transcript: "hello",
          is_low_confidence: false,
          low_confidence_sections: [],
          confidence: 1,
        }),
      ),
    );

    const result = await parseAssessment(new File([], "session.wav"));
    expect(result.transcript).toBe("hello");
    expect(result.assessment.clinicalDetails.chiefComplaint).toBe("Knee pain");
  });

  it("throws ApiShapeError when the response is missing required fields", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        jsonResponse({
          assessment: validAssessment,
          // transcript and confidence fields missing -- simulates backend drift
        }),
      ),
    );

    await expect(parseAssessment(new File([], "session.wav"))).rejects.toBeInstanceOf(
      ApiShapeError,
    );
  });

  it("throws ApiError with the response detail on a non-2xx status", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(jsonResponse({ detail: "Only .wav files are supported" }, 400)),
    );

    await expect(parseAssessment(new File([], "session.mp3"))).rejects.toMatchObject({
      status: 400,
    });
  });
});

describe("listAssessments", () => {
  it("rejects a malformed array element instead of silently returning bad data", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        jsonResponse([{ ...validAssessment, id: "1" /* missing createdAt */ }]),
      ),
    );

    await expect(listAssessments()).rejects.toBeInstanceOf(ApiShapeError);
  });
});

describe("getAssessment", () => {
  it("returns a valid saved assessment", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        jsonResponse({ ...validAssessment, id: "abc123", createdAt: "2026-01-01T00:00:00Z" }),
      ),
    );

    const result = await getAssessment("abc123");
    expect(result.id).toBe("abc123");
  });

  it("throws ApiError on 404", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(jsonResponse({ detail: "Assessment not found" }, 404)),
    );

    const err = await getAssessment("missing").catch((e) => e);
    expect(err).toBeInstanceOf(ApiError);
    expect((err as ApiError).status).toBe(404);
  });
});
