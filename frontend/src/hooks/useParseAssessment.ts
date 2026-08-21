import { useCallback, useState } from "react";
import { ApiError, ApiShapeError, parseAssessment } from "../api";
import type { ParseDebugResult } from "../types";

export type ParseStatus = "idle" | "parsing" | "parsed" | "error";

interface State {
  status: ParseStatus;
  result: ParseDebugResult | null;
  error: string;
}

export function useParseAssessment() {
  const [state, setState] = useState<State>({ status: "idle", result: null, error: "" });

  const parse = useCallback(async (file: File) => {
    setState({ status: "parsing", result: null, error: "" });
    try {
      const result = await parseAssessment(file);
      setState({ status: "parsed", result, error: "" });
    } catch (err) {
      let message = "Could not reach the API. Is the FastAPI server running?";
      if (err instanceof ApiError) {
        message = typeof err.detail === "string" ? err.detail : "Failed to parse audio";
      } else if (err instanceof ApiShapeError) {
        message = err.message;
      }
      setState({ status: "error", result: null, error: message });
    }
  }, []);

  const reset = useCallback(() => {
    setState({ status: "idle", result: null, error: "" });
  }, []);

  return { ...state, parse, reset };
}
