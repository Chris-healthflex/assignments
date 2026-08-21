import { NavLink, Outlet } from "react-router-dom";

function App() {
  return (
    <div className="min-h-screen">
      <header className="border-b border-slate-200 bg-white">
        <div className="mx-auto max-w-4xl px-6 py-5">
          <p className="text-xs font-medium uppercase tracking-wide text-teal-600">
            Stance Health
          </p>
          <h1 className="text-xl font-semibold text-slate-900">
            Clinical Assessment Pipeline
          </h1>
        </div>
      </header>

      <nav className="border-b border-slate-200 bg-white">
        <div className="mx-auto flex max-w-4xl gap-6 px-6">
          <NavLink
            to="/"
            end
            className={({ isActive }) =>
              `border-b-2 py-3 text-sm font-medium transition ${
                isActive
                  ? "border-teal-600 text-teal-700"
                  : "border-transparent text-slate-500 hover:text-slate-700"
              }`
            }
          >
            New Assessment
          </NavLink>
          <NavLink
            to="/history"
            className={({ isActive }) =>
              `border-b-2 py-3 text-sm font-medium transition ${
                isActive
                  ? "border-teal-600 text-teal-700"
                  : "border-transparent text-slate-500 hover:text-slate-700"
              }`
            }
          >
            Saved Assessments
          </NavLink>
        </div>
      </nav>

      <main className="mx-auto max-w-4xl px-6 py-8">
        <Outlet />
      </main>
    </div>
  );
}

export default App;
