import { useCallback, useRef, useState } from "react";
import { saveAssessment } from "../api";
import type { FirstAssessment } from "../types";

export type SaveStatus = "idle" | "saving" | "saved" | "error";

export function useSaveAssessment() {
  const [status, setStatus] = useState<SaveStatus>("idle");
  const [savedId, setSavedId] = useState<string>("");
  const isSavingRef = useRef(false);

  const save = useCallback(async (assessment: FirstAssessment) => {
    if (isSavingRef.current) return;
    isSavingRef.current = true;
    setStatus("saving");
    try {
      const { id } = await saveAssessment(assessment);
      setSavedId(id);
      setStatus("saved");
    } catch {
      setStatus("error");
    } finally {
      isSavingRef.current = false;
    }
  }, []);

  const reset = useCallback(() => {
    setStatus("idle");
    setSavedId("");
    isSavingRef.current = false;
  }, []);

  return { status, savedId, save, reset };
}
