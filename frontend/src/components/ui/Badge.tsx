import type { HTMLAttributes } from "react";

type Tone = "teal" | "amber" | "slate";

const toneClasses: Record<Tone, string> = {
  teal: "bg-teal-100 text-teal-800 dark:bg-teal-900/60 dark:text-teal-200",
  amber: "bg-amber-100 text-amber-800 dark:bg-amber-900/60 dark:text-amber-200",
  slate: "bg-slate-100 text-slate-700 dark:bg-slate-800 dark:text-slate-300",
};

interface Props extends HTMLAttributes<HTMLSpanElement> {
  tone?: Tone;
}

export function Badge({ tone = "slate", className = "", ...props }: Props) {
  return (
    <span
      className={`rounded-full px-2.5 py-0.5 text-xs font-medium ${toneClasses[tone]} ${className}`}
      {...props}
    />
  );
}
