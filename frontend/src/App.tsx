import { NavLink, Outlet, useNavigate } from "react-router-dom";
import { useCommands, useRegisterCommands } from "./components/CommandPalette";
import { useTheme } from "./hooks/useTheme";

const navLinkClass = ({ isActive }: { isActive: boolean }) =>
  `border-b-2 py-3 text-sm font-medium transition ${
    isActive
      ? "border-teal-600 text-teal-700 dark:border-teal-400 dark:text-teal-400"
      : "border-transparent text-slate-500 hover:text-slate-700 dark:text-slate-400 dark:hover:text-slate-200"
  }`;

function App() {
  const navigate = useNavigate();
  const { theme, toggle } = useTheme();
  const { open: openPalette } = useCommands();

  useRegisterCommands(
    [
      {
        id: "nav-new",
        group: "Go to",
        label: "New Assessment",
        run: () => navigate("/"),
      },
      {
        id: "nav-history",
        group: "Go to",
        label: "Saved Assessments",
        run: () => navigate("/history"),
      },
      {
        id: "toggle-theme",
        group: "View",
        label: theme === "dark" ? "Switch to light mode" : "Switch to dark mode",
        run: toggle,
      },
    ],
    [navigate, theme, toggle],
  );

  return (
    <div className="min-h-screen">
      <header className="border-b border-slate-200 bg-white dark:border-slate-800 dark:bg-slate-900">
        <div className="mx-auto flex max-w-5xl items-center justify-between gap-4 px-6 py-5">
          <div>
            <p className="text-xs font-medium uppercase tracking-wide text-teal-600 dark:text-teal-400">
              Stance Health
            </p>
            <h1 className="text-xl font-semibold text-slate-900 dark:text-slate-100">
              Clinical Assessment Pipeline
            </h1>
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={openPalette}
              aria-label="Open command palette"
              className="hidden items-center gap-2 rounded-lg border border-slate-200 px-3 py-1.5 text-xs text-slate-500 transition hover:border-slate-300 hover:text-slate-700 sm:flex dark:border-slate-700 dark:text-slate-400 dark:hover:text-slate-200"
            >
              Search commands
              <kbd className="rounded border border-slate-200 px-1.5 py-px font-mono text-[10px] dark:border-slate-700">
                ⌘K
              </kbd>
            </button>

            <button
              onClick={toggle}
              aria-label={
                theme === "dark" ? "Switch to light mode" : "Switch to dark mode"
              }
              className="rounded-lg border border-slate-200 p-2 text-slate-500 transition hover:text-slate-800 dark:border-slate-700 dark:text-slate-400 dark:hover:text-slate-100"
            >
              {theme === "dark" ? <SunIcon /> : <MoonIcon />}
            </button>
          </div>
        </div>
      </header>

      <nav className="border-b border-slate-200 bg-white dark:border-slate-800 dark:bg-slate-900">
        <div className="mx-auto flex max-w-5xl gap-6 px-6">
          <NavLink to="/" end className={navLinkClass}>
            New Assessment
          </NavLink>
          <NavLink to="/history" className={navLinkClass}>
            Saved Assessments
          </NavLink>
        </div>
      </nav>

      <main className="mx-auto max-w-5xl px-6 py-8">
        <Outlet />
      </main>
    </div>
  );
}

function SunIcon() {
  return (
    <svg viewBox="0 0 24 24" className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth={2}>
      <circle cx="12" cy="12" r="4" />
      <path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4" />
    </svg>
  );
}

function MoonIcon() {
  return (
    <svg viewBox="0 0 24 24" className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth={2}>
      <path d="M21 12.8A9 9 0 1111.2 3a7 7 0 009.8 9.8z" />
    </svg>
  );
}

export default App;
