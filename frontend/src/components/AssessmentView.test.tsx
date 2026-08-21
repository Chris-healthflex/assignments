import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { AssessmentView } from "./AssessmentView";
import type { FirstAssessment } from "../types";

function emptyAssessment(): FirstAssessment {
  return {
    clinicalDetails: { clinicalHistory: "", chiefComplaint: "", duration: "" },
    subjectiveAssessments: [],
    objectiveAssessment: { tests: [] },
    subjectiveGoals: [],
    objectiveGoals: [],
    recommendation: [],
    patientAdvice: { adviceDetails: "" },
  };
}

describe("AssessmentView", () => {
  it("renders read-only text for populated fields by default", () => {
    const assessment = {
      ...emptyAssessment(),
      clinicalDetails: {
        clinicalHistory: "history",
        chiefComplaint: "Knee pain",
        duration: "8 months",
      },
    };

    render(<AssessmentView assessment={assessment} />);

    expect(screen.getByText("Knee pain")).toBeInTheDocument();
    expect(screen.queryByRole("textbox")).not.toBeInTheDocument();
  });

  it("shows a flagged badge and 'None recorded' for empty sections when read-only", () => {
    render(
      <AssessmentView
        assessment={emptyAssessment()}
        flaggedSections={["subjectiveAssessments"]}
      />,
    );

    expect(screen.getByText("Not stated in the recording")).toBeInTheDocument();
    expect(screen.getAllByText("None recorded").length).toBeGreaterThan(0);
  });

  it("renders editable inputs and calls onChange when a field is edited", async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();
    const assessment = emptyAssessment();

    render(<AssessmentView assessment={assessment} editable onChange={onChange} />);

    const chiefComplaintInput = screen.getByLabelText("Chief Complaint");
    await user.type(chiefComplaintInput, "K");

    expect(onChange).toHaveBeenCalledWith(
      expect.objectContaining({
        clinicalDetails: expect.objectContaining({ chiefComplaint: "K" }),
      }),
    );
  });

  it("adds a blank item and calls onChange when '+ Add manually' is clicked", async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();

    render(
      <AssessmentView
        assessment={emptyAssessment()}
        flaggedSections={["subjectiveGoals"]}
        editable
        onChange={onChange}
      />,
    );

    const addButtons = screen.getAllByRole("button", { name: "+ Add manually" });
    await user.click(addButtons[0]);

    expect(onChange).toHaveBeenCalled();
    const updated = onChange.mock.calls[0][0] as FirstAssessment;
    const addedList =
      updated.subjectiveAssessments.length > 0
        ? updated.subjectiveAssessments
        : updated.subjectiveGoals;
    expect(addedList.length).toBe(1);
  });

  describe("evidence", () => {
    function withTests(): FirstAssessment {
      return {
        ...emptyAssessment(),
        objectiveAssessment: {
          tests: [
            {
              testName: "Knee flexion",
              unitName: "degrees",
              value: "",
              left: "124",
              right: "130",
              comments: "",
            },
            {
              testName: "Ankle dorsiflexion",
              unitName: "degrees",
              value: "",
              left: "4.5",
              right: "12",
              comments: "",
            },
          ],
        },
      };
    }

    it("qualifies a citation with the row it belongs to", async () => {
      const user = userEvent.setup();
      const onInspectField = vi.fn();

      render(
        <AssessmentView
          assessment={withTests()}
          evidence={[
            { field: "objectiveAssessment.tests[1].left", segmentIds: [13], quote: "4.5" },
          ]}
          onInspectField={onInspectField}
        />,
      );

      // Five rows all have a field labelled "Left"; the citation says which.
      await user.click(
        screen.getByRole("button", {
          name: "Show transcript evidence for Ankle dorsiflexion · Left",
        }),
      );

      expect(onInspectField).toHaveBeenCalledWith(
        "objectiveAssessment.tests[1].left",
        "Ankle dorsiflexion · Left",
        [13],
      );
    });

    it("treats a citation on a parent path as covering its children", () => {
      render(
        <AssessmentView
          assessment={withTests()}
          evidence={[
            { field: "objectiveAssessment.tests[0]", segmentIds: [10], quote: "flexion 124" },
          ]}
          onInspectField={vi.fn()}
        />,
      );

      expect(
        screen.getByRole("button", {
          name: "Show transcript evidence for Knee flexion · Unit",
        }),
      ).toBeInTheDocument();
    });

    it("marks a populated but uncited field as unverified", () => {
      render(
        <AssessmentView
          assessment={withTests()}
          evidence={[]}
          ungroundedFields={["objectiveAssessment.tests[0].right"]}
          onInspectField={vi.fn()}
        />,
      );

      expect(
        screen.getByLabelText("Knee flexion · Right has no transcript evidence"),
      ).toBeInTheDocument();
      expect(
        screen.queryByLabelText("Knee flexion · Left has no transcript evidence"),
      ).not.toBeInTheDocument();
    });

    it("shows no evidence affordances at all when none was supplied", () => {
      render(<AssessmentView assessment={withTests()} />);

      expect(screen.queryByText(/⏱/)).not.toBeInTheDocument();
      expect(screen.queryByText("⚠ unverified")).not.toBeInTheDocument();
    });
  });
});
