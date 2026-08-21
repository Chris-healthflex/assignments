import type { ButtonHTMLAttributes } from "react";

type Variant = "primary" | "secondary" | "ghost" | "dark";

const variantClasses: Record<Variant, string> = {
  primary: "bg-teal-600 text-white hover:bg-teal-700",
  dark: "bg-slate-900 text-white hover:bg-slate-700 dark:bg-slate-100 dark:text-slate-900 dark:hover:bg-white",
  secondary:
    "border border-slate-200 bg-white text-slate-700 hover:border-teal-300 shadow-sm dark:border-slate-700 dark:bg-slate-900 dark:text-slate-200 dark:hover:border-teal-700",
  ghost: "text-slate-500 hover:text-slate-700 dark:text-slate-400 dark:hover:text-slate-200",
};

interface Props extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: Variant;
}

export function Button({ variant = "primary", className = "", ...props }: Props) {
  return (
    <button
      className={`rounded-lg px-4 py-2 text-sm font-medium transition disabled:opacity-50 ${variantClasses[variant]} ${className}`}
      {...props}
    />
  );
}
