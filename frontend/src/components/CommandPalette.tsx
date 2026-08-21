import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react";

export interface Command {
  id: string;
  label: string;
  hint?: string;
  group?: string;
  run: () => void;
}

interface CommandStore {
  commands: Command[];
  register: (commands: Command[]) => () => void;
  open: () => void;
}

const CommandContext = createContext<CommandStore | null>(null);

export function useCommands(): CommandStore {
  const store = useContext(CommandContext);
  if (!store) throw new Error("useCommands must be used inside <CommandProvider>");
  return store;
}

/**
 * Lets a page contribute commands for as long as it is mounted.
 *
 * Pages own actions the shell knows nothing about ("Save to MongoDB", "Export
 * JSON"), so rather than centralising a list that drifts out of sync, each page
 * registers what it can currently do and withdraws it on unmount.
 */
export function useRegisterCommands(commands: Command[], deps: unknown[] = []) {
  const { register } = useCommands();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  const memoized = useMemo(() => commands, deps);

  useEffect(() => register(memoized), [register, memoized]);
}

export function CommandProvider({ children }: { children: ReactNode }) {
  const [registrations, setRegistrations] = useState<Command[][]>([]);
  const [isOpen, setIsOpen] = useState(false);

  const register = useCallback((commands: Command[]) => {
    setRegistrations((current) => [...current, commands]);
    return () =>
      setRegistrations((current) => current.filter((entry) => entry !== commands));
  }, []);

  const commands = useMemo(() => registrations.flat(), [registrations]);

  useEffect(() => {
    const handler = (event: KeyboardEvent) => {
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "k") {
        event.preventDefault();
        setIsOpen((open) => !open);
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, []);

  const store = useMemo<CommandStore>(
    () => ({ commands, register, open: () => setIsOpen(true) }),
    [commands, register],
  );

  return (
    <CommandContext.Provider value={store}>
      {children}
      {isOpen && (
        <Palette commands={commands} onClose={() => setIsOpen(false)} />
      )}
    </CommandContext.Provider>
  );
}

function Palette({
  commands,
  onClose,
}: {
  commands: Command[];
  onClose: () => void;
}) {
  const [query, setQuery] = useState("");
  const [activeIndex, setActiveIndex] = useState(0);

  const matches = useMemo(() => {
    const needle = query.trim().toLowerCase();
    if (!needle) return commands;
    return commands.filter((command) =>
      `${command.group ?? ""} ${command.label}`.toLowerCase().includes(needle),
    );
  }, [commands, query]);

  useEffect(() => setActiveIndex(0), [query]);

  function runAt(index: number) {
    const command = matches[index];
    if (!command) return;
    onClose();
    command.run();
  }

  function onKeyDown(event: React.KeyboardEvent) {
    if (event.key === "Escape") return onClose();
    if (event.key === "ArrowDown") {
      event.preventDefault();
      setActiveIndex((i) => Math.min(i + 1, matches.length - 1));
    }
    if (event.key === "ArrowUp") {
      event.preventDefault();
      setActiveIndex((i) => Math.max(i - 1, 0));
    }
    if (event.key === "Enter") {
      event.preventDefault();
      runAt(activeIndex);
    }
  }

  return (
    <div
      className="fixed inset-0 z-50 flex items-start justify-center bg-slate-900/40 p-4 pt-[15vh] backdrop-blur-sm"
      onClick={onClose}
    >
      <div
        role="dialog"
        aria-modal="true"
        aria-label="Command palette"
        onClick={(e) => e.stopPropagation()}
        className="w-full max-w-lg overflow-hidden rounded-xl border border-slate-200 bg-white shadow-2xl dark:border-slate-700 dark:bg-slate-900"
      >
        <input
          autoFocus
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={onKeyDown}
          placeholder="Type a command…"
          aria-label="Search commands"
          className="w-full border-b border-slate-200 px-4 py-3 text-sm text-slate-800 outline-none dark:border-slate-700 dark:bg-slate-900 dark:text-slate-100"
        />

        <ul className="max-h-72 overflow-y-auto py-1">
          {matches.length === 0 && (
            <li className="px-4 py-6 text-center text-sm text-slate-400">
              No matching commands
            </li>
          )}
          {matches.map((command, index) => (
            <li key={command.id}>
              <button
                onMouseEnter={() => setActiveIndex(index)}
                onClick={() => runAt(index)}
                className={`flex w-full items-center justify-between gap-3 px-4 py-2 text-left text-sm transition ${
                  index === activeIndex
                    ? "bg-teal-50 text-teal-900 dark:bg-teal-950/60 dark:text-teal-200"
                    : "text-slate-700 dark:text-slate-300"
                }`}
              >
                <span>
                  {command.group && (
                    <span className="mr-2 text-xs text-slate-400">
                      {command.group}
                    </span>
                  )}
                  {command.label}
                </span>
                {command.hint && (
                  <kbd className="shrink-0 rounded border border-slate-200 px-1.5 py-px font-mono text-[10px] text-slate-500 dark:border-slate-700">
                    {command.hint}
                  </kbd>
                )}
              </button>
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}
