import type { HTMLAttributes } from "react";

interface Props extends HTMLAttributes<HTMLDivElement> {
  tone?: "default" | "flagged";
}

export function Card({ tone = "default", className = "", ...props }: Props) {
  const toneClasses =
    tone === "flagged"
      ? "border-amber-300 bg-amber-50/40 dark:border-amber-800 dark:bg-amber-950/20"
      : "border-slate-200 bg-white dark:border-slate-800 dark:bg-slate-900";

  return (
    <div
      className={`rounded-xl border shadow-sm ${toneClasses} ${className}`}
      {...props}
    />
  );
}
