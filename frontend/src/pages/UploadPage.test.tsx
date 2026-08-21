import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { UploadPage } from "./UploadPage";
import { ToastProvider } from "../components/ui/Toast";
import * as api from "../api";
import type { ParseDebugResult } from "../types";

function renderUploadPage() {
  return render(
    <ToastProvider>
      <UploadPage />
    </ToastProvider>,
  );
}

function fakeParseResult(overrides: Partial<ParseDebugResult> = {}): ParseDebugResult {
  return {
    assessment: {
      clinicalDetails: { clinicalHistory: "", chiefComplaint: "Knee pain", duration: "" },
      subjectiveAssessments: [],
      objectiveAssessment: { tests: [] },
      subjectiveGoals: [],
      objectiveGoals: [],
      recommendation: [],
      patientAdvice: { adviceDetails: "" },
    },
    transcript: "Patient reports knee pain.",
    is_low_confidence: false,
    low_confidence_sections: [],
    confidence: 1,
    ...overrides,
  };
}

async function uploadFile(file = new File(["wav-bytes"], "session.wav", { type: "audio/wav" })) {
  const input = document.querySelector('input[type="file"]') as HTMLInputElement;
  const user = userEvent.setup();
  await user.upload(input, file);
  return user;
}

describe("UploadPage", () => {
  it("shows the idle empty state before any file is chosen", () => {
    renderUploadPage();
    expect(
      screen.getByText("Turn a session recording into a structured assessment"),
    ).toBeInTheDocument();
    expect(screen.getByText("1. Upload")).toBeInTheDocument();
  });

  it("renders the extracted assessment and confidence badge after a successful parse", async () => {
    vi.spyOn(api, "parseAssessment").mockResolvedValue(fakeParseResult());

    renderUploadPage();
    await uploadFile();

    await waitFor(() => {
      expect(screen.getByText("Extracted Assessment")).toBeInTheDocument();
    });
    expect(screen.getByText("100%")).toBeInTheDocument();
    expect(screen.getByDisplayValue("Knee pain")).toBeInTheDocument();
  });

  it("shows an error message when parsing fails", async () => {
    vi.spyOn(api, "parseAssessment").mockRejectedValue(
      new api.ApiError(400, "Only .wav files are supported"),
    );

    renderUploadPage();
    await uploadFile();

    await waitFor(() => {
      expect(screen.getByText("Only .wav files are supported")).toBeInTheDocument();
    });
  });

  it("saves the assessment and shows a success toast", async () => {
    vi.spyOn(api, "parseAssessment").mockResolvedValue(fakeParseResult());
    vi.spyOn(api, "saveAssessment").mockResolvedValue({ id: "saved-id-123" });

    renderUploadPage();
    const user = await uploadFile();

    const saveButton = await screen.findByRole("button", { name: "Save to MongoDB" });
    await user.click(saveButton);

    await waitFor(() => {
      expect(screen.getByText(/Saved · id saved-id-123/)).toBeInTheDocument();
    });
    expect(screen.getByRole("status")).toHaveTextContent("Assessment saved to MongoDB");
  });

  it("highlights flagged sections and lets the reviewer complete them before saving", async () => {
    vi.spyOn(api, "parseAssessment").mockResolvedValue(
      fakeParseResult({
        is_low_confidence: true,
        low_confidence_sections: ["patientAdvice"],
        confidence: 0.86,
      }),
    );

    renderUploadPage();
    await uploadFile();

    await waitFor(() => {
      expect(screen.getByText("Not stated in the recording")).toBeInTheDocument();
    });
    expect(
      screen.getByText(/Sections highlighted below weren't covered/),
    ).toBeInTheDocument();
  });
});
