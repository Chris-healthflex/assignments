import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import {
  CommandProvider,
  useCommands,
  useRegisterCommands,
  type Command,
} from "./CommandPalette";

function Consumer({ commands }: { commands: Command[] }) {
  const { open } = useCommands();
  useRegisterCommands(commands, [commands]);
  return <button onClick={open}>Open palette</button>;
}

function renderPalette(commands: Command[]) {
  render(
    <CommandProvider>
      <Consumer commands={commands} />
    </CommandProvider>,
  );
  return userEvent.setup();
}

describe("CommandPalette", () => {
  it("opens on the toolbar button and lists registered commands", async () => {
    const user = renderPalette([
      { id: "a", group: "Go to", label: "Saved Assessments", run: vi.fn() },
    ]);

    await user.click(screen.getByText("Open palette"));

    expect(screen.getByRole("dialog", { name: "Command palette" })).toBeInTheDocument();
    expect(screen.getByText("Saved Assessments")).toBeInTheDocument();
  });

  it("opens on Cmd+K", async () => {
    const user = renderPalette([{ id: "a", label: "Save to MongoDB", run: vi.fn() }]);

    await user.keyboard("{Meta>}k{/Meta}");

    expect(screen.getByRole("dialog", { name: "Command palette" })).toBeInTheDocument();
  });

  it("filters commands by the typed query", async () => {
    const user = renderPalette([
      { id: "a", label: "Save to MongoDB", run: vi.fn() },
      { id: "b", label: "Download as JSON", run: vi.fn() },
    ]);

    await user.click(screen.getByText("Open palette"));
    await user.type(screen.getByLabelText("Search commands"), "json");

    expect(screen.getByText("Download as JSON")).toBeInTheDocument();
    expect(screen.queryByText("Save to MongoDB")).not.toBeInTheDocument();
  });

  it("runs the highlighted command on Enter and closes", async () => {
    const download = vi.fn();
    const user = renderPalette([
      { id: "a", label: "Save to MongoDB", run: vi.fn() },
      { id: "b", label: "Download as JSON", run: download },
    ]);

    await user.click(screen.getByText("Open palette"));
    await user.keyboard("{ArrowDown}{Enter}");

    expect(download).toHaveBeenCalledOnce();
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  });

  it("closes on Escape without running anything", async () => {
    const run = vi.fn();
    const user = renderPalette([{ id: "a", label: "Save to MongoDB", run }]);

    await user.click(screen.getByText("Open palette"));
    await user.keyboard("{Escape}");

    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
    expect(run).not.toHaveBeenCalled();
  });

  it("tells the user when nothing matches", async () => {
    const user = renderPalette([{ id: "a", label: "Save to MongoDB", run: vi.fn() }]);

    await user.click(screen.getByText("Open palette"));
    await user.type(screen.getByLabelText("Search commands"), "zzz");

    expect(screen.getByText("No matching commands")).toBeInTheDocument();
  });
});
