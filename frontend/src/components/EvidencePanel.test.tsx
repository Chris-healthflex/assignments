import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { EvidencePanel, formatTime } from "./EvidencePanel";
import type { TranscriptSegment } from "../types";

const segments: TranscriptSegment[] = [
  { id: 0, start: 0, end: 5, text: "Patient reports right knee pain." },
  { id: 1, start: 5, end: 12, text: "Flexion measured at 110 degrees." },
  { id: 2, start: 12, end: 20, text: "We'll start with quad strengthening." },
];

function renderPanel(props: Partial<Parameters<typeof EvidencePanel>[0]> = {}) {
  const onClearCitation = vi.fn();
  render(
    <EvidencePanel
      segments={segments}
      audioUrl="blob:mock"
      citedSegmentIds={[]}
      citedFieldLabel={null}
      onClearCitation={onClearCitation}
      {...props}
    />,
  );
  return { onClearCitation };
}

describe("formatTime", () => {
  it("renders m:ss and pads the seconds", () => {
    expect(formatTime(0)).toBe("0:00");
    expect(formatTime(9.4)).toBe("0:09");
    expect(formatTime(75)).toBe("1:15");
  });

  it("falls back to zero for values an unloaded <audio> reports", () => {
    expect(formatTime(NaN)).toBe("0:00");
    expect(formatTime(Infinity)).toBe("0:00");
    expect(formatTime(-3)).toBe("0:00");
  });
});

describe("EvidencePanel", () => {
  it("lists every transcript segment with its timestamp", () => {
    renderPanel();
    expect(screen.getByText("Patient reports right knee pain.")).toBeInTheDocument();
    expect(screen.getByText("0:05")).toBeInTheDocument();
    expect(screen.getByText("0:12")).toBeInTheDocument();
  });

  it("seeks the audio to a segment when it is clicked", async () => {
    renderPanel();
    const audio = document.querySelector("audio") as HTMLAudioElement;

    await userEvent.click(screen.getByText("Flexion measured at 110 degrees."));

    expect(audio.currentTime).toBe(5);
  });

  it("announces which field is being verified and offers a way out", async () => {
    const { onClearCitation } = renderPanel({
      citedSegmentIds: [1],
      citedFieldLabel: "Flexion",
    });

    expect(screen.getByText("Flexion")).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "Clear" }));
    expect(onClearCitation).toHaveBeenCalled();
  });

  it("jumps the audio to the first cited segment", () => {
    renderPanel({ citedSegmentIds: [2], citedFieldLabel: "Recommendation" });
    const audio = document.querySelector("audio") as HTMLAudioElement;
    expect(audio.currentTime).toBe(12);
  });

  it("says so when the model cited nothing for the selected field", () => {
    renderPanel({ citedSegmentIds: [], citedFieldLabel: "Patient Advice" });
    expect(screen.getByText(/the model cited no segment/)).toBeInTheDocument();
  });

  it("still shows the transcript when there is no audio to play", () => {
    renderPanel({ audioUrl: null });
    expect(document.querySelector("audio")).toBeNull();
    expect(screen.getByText("Patient reports right knee pain.")).toBeInTheDocument();
  });
});
