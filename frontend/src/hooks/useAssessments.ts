import { useCallback, useEffect, useState } from "react";
import { listAssessments } from "../api";
import type { SavedAssessment } from "../types";

interface State {
  data: SavedAssessment[];
  loading: boolean;
  error: string | null;
}

export function useAssessments() {
  const [state, setState] = useState<State>({ data: [], loading: true, error: null });

  const refetch = useCallback(async () => {
    setState((s) => ({ ...s, loading: true, error: null }));
    try {
      const data = await listAssessments();
      setState({ data, loading: false, error: null });
    } catch {
      setState({ data: [], loading: false, error: "Could not load saved assessments" });
    }
  }, []);

  useEffect(() => {
    refetch();
  }, [refetch]);

  return { ...state, refetch };
}
