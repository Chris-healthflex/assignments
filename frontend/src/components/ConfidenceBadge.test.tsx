import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";
import { ConfidenceBadge } from "./ConfidenceBadge";

describe("ConfidenceBadge", () => {
  it("renders confidence as a rounded percentage", () => {
    render(<ConfidenceBadge confidence={0.428} flaggedCount={4} />);
    expect(screen.getByText("43%")).toBeInTheDocument();
  });

  it("shows the flagged section count when there are flagged sections", () => {
    render(<ConfidenceBadge confidence={0.43} flaggedCount={4} />);
    expect(
      screen.getByText("4 of 7 sections had no supporting content in the recording."),
    ).toBeInTheDocument();
  });

  it("hides the flagged count line when nothing is flagged", () => {
    render(<ConfidenceBadge confidence={1} flaggedCount={0} />);
    expect(screen.queryByText(/sections had no supporting content/)).not.toBeInTheDocument();
  });

  it("reveals the formula only after clicking the explainer toggle", async () => {
    const user = userEvent.setup();
    render(<ConfidenceBadge confidence={0.9} flaggedCount={1} />);

    expect(screen.queryByText(/coverage measure/)).not.toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "How is this calculated?" }));

    expect(screen.getByText(/coverage measure/)).toBeInTheDocument();
  });
});
