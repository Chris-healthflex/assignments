import { useCallback, useEffect, useState } from "react";
import { listAssessments, type AssessmentFilters } from "../api";
import type { SavedAssessment } from "../types";

interface State {
  data: SavedAssessment[];
  loading: boolean;
  error: string | null;
}

export function useAssessments(filters: AssessmentFilters = {}) {
  const [state, setState] = useState<State>({ data: [], loading: true, error: null });
  const { dateFrom, dateTo } = filters;

  const refetch = useCallback(async () => {
    setState((s) => ({ ...s, loading: true, error: null }));
    try {
      const data = await listAssessments({ dateFrom, dateTo });
      setState({ data, loading: false, error: null });
    } catch {
      setState({ data: [], loading: false, error: "Could not load saved assessments" });
    }
  }, [dateFrom, dateTo]);

  useEffect(() => {
    refetch();
  }, [refetch]);

  return { ...state, refetch };
}
