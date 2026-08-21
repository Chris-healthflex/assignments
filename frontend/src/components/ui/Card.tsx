import type { HTMLAttributes } from "react";

interface Props extends HTMLAttributes<HTMLDivElement> {
  tone?: "default" | "flagged";
}

export function Card({ tone = "default", className = "", ...props }: Props) {
  const toneClasses =
    tone === "flagged"
      ? "border-amber-300 bg-amber-50/40"
      : "border-slate-200 bg-white";

  return (
    <div
      className={`rounded-xl border shadow-sm ${toneClasses} ${className}`}
      {...props}
    />
  );
}
