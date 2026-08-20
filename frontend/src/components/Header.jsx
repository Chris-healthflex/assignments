import { useEffect, useState } from "react";

import { checkHealth } from "../lib/api";

const VIEWS = [
  { id: "new", label: "New" },
  { id: "browse", label: "Browse" },
];

export function Header({ view, onView }) {
  const [health, setHealth] = useState(null);

  useEffect(() => {
    let live = true;
    const poll = async () => {
      const state = await checkHealth();
      if (live) setHealth(state);
    };
    poll();
    const timer = setInterval(poll, 30000);
    return () => {
      live = false;
      clearInterval(timer);
    };
  }, []);

  const tone = !health ? "" : health.mongo ? "ok" : "down";
  const label = !health
    ? "checking..."
    : !health.reachable
      ? "api unreachable"
      : health.mongo
        ? "database connected"
        : "database unreachable";

  return (
    <header className="app-header">
      <h1>
        First Assessment <span>clinical extraction</span>
      </h1>
      <nav>
        {VIEWS.map((item) => (
          <button
            key={item.id}
            type="button"
            aria-current={view === item.id}
            onClick={() => onView(item.id)}
          >
            {item.label}
          </button>
        ))}
      </nav>
      <span className={`health ${tone}`}>
        <b aria-hidden="true" />
        {label}
      </span>
    </header>
  );
}
