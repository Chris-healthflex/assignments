import { useEffect, useState } from "react";
import { getAssessment } from "../api";
import type { SavedAssessment } from "../types";

interface State {
  data: SavedAssessment | null;
  loading: boolean;
  error: string | null;
}

export function useAssessment(id: string | undefined) {
  const [state, setState] = useState<State>({ data: null, loading: true, error: null });

  useEffect(() => {
    if (!id) {
      setState({ data: null, loading: false, error: "No assessment id given" });
      return;
    }

    let cancelled = false;
    setState({ data: null, loading: true, error: null });

    getAssessment(id)
      .then((data) => {
        if (!cancelled) setState({ data, loading: false, error: null });
      })
      .catch(() => {
        if (!cancelled) {
          setState({ data: null, loading: false, error: "Could not load this assessment" });
        }
      });

    return () => {
      cancelled = true;
    };
  }, [id]);

  return state;
}
