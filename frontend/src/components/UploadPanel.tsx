import { useRef, useState } from "react";
import { ApiError, isParseErrorDetail, parseAssessment, saveAssessment } from "../api";
import type { FirstAssessment } from "../types";
import { AssessmentView } from "./AssessmentView";

type Status = "idle" | "parsing" | "parsed" | "saving" | "saved" | "error";

export function UploadPanel() {
  const [status, setStatus] = useState<Status>("idle");
  const [fileName, setFileName] = useState<string>("");
  const [assessment, setAssessment] = useState<FirstAssessment | null>(null);
  const [error, setError] = useState<string>("");
  const [lowConfidenceSections, setLowConfidenceSections] = useState<string[]>([]);
  const [savedId, setSavedId] = useState<string>("");
  const inputRef = useRef<HTMLInputElement>(null);

  async function handleFile(file: File) {
    setFileName(file.name);
    setStatus("parsing");
    setError("");
    setLowConfidenceSections([]);
    setAssessment(null);
    setSavedId("");

    try {
      const result = await parseAssessment(file);
      setAssessment(result);
      setStatus("parsed");
    } catch (err) {
      if (err instanceof ApiError && err.status === 422 && isParseErrorDetail(err.detail)) {
        setError(err.detail.message);
        setLowConfidenceSections(err.detail.low_confidence_sections);
      } else if (err instanceof ApiError) {
        setError(typeof err.detail === "string" ? err.detail : "Failed to parse audio");
      } else {
        setError("Could not reach the API. Is the FastAPI server running?");
      }
      setStatus("error");
    }
  }

  async function handleSave() {
    if (!assessment) return;
    setStatus("saving");
    try {
      const { id } = await saveAssessment(assessment);
      setSavedId(id);
      setStatus("saved");
    } catch {
      setError("Failed to save assessment");
      setStatus("error");
    }
  }

  return (
    <div className="space-y-6">
      <div
        className="flex flex-col items-center justify-center rounded-2xl border-2 border-dashed border-slate-300 bg-white p-10 text-center transition hover:border-teal-400"
        onDragOver={(e) => e.preventDefault()}
        onDrop={(e) => {
          e.preventDefault();
          const file = e.dataTransfer.files?.[0];
          if (file) handleFile(file);
        }}
      >
        <p className="mb-1 text-sm font-medium text-slate-700">
          Drop a clinical session WAV file here
        </p>
        <p className="mb-4 text-xs text-slate-500">or</p>
        <button
          onClick={() => inputRef.current?.click()}
          className="rounded-lg bg-teal-600 px-4 py-2 text-sm font-medium text-white shadow-sm transition hover:bg-teal-700"
        >
          Choose file
        </button>
        <input
          ref={inputRef}
          type="file"
          accept=".wav,audio/wav"
          className="hidden"
          onChange={(e) => {
            const file = e.target.files?.[0];
            if (file) handleFile(file);
          }}
        />
        {fileName && <p className="mt-4 text-xs text-slate-500">{fileName}</p>}
      </div>

      {status === "parsing" && (
        <p className="text-center text-sm text-slate-500">
          Transcribing and extracting clinical data…
        </p>
      )}

      {status === "error" && (
        <div className="rounded-xl border border-amber-300 bg-amber-50 p-4 text-sm text-amber-900">
          <p className="font-medium">{error || "Something went wrong"}</p>
          {lowConfidenceSections.length > 0 && (
            <p className="mt-1">
              Low-confidence sections: {lowConfidenceSections.join(", ")}
            </p>
          )}
        </div>
      )}

      {assessment && (
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="text-lg font-semibold text-slate-900">Extracted Assessment</h2>
            {status === "saved" ? (
              <span className="rounded-full bg-teal-100 px-3 py-1 text-xs font-medium text-teal-800">
                Saved · id {savedId}
              </span>
            ) : (
              <button
                onClick={handleSave}
                disabled={status === "saving"}
                className="rounded-lg bg-slate-900 px-4 py-2 text-sm font-medium text-white shadow-sm transition hover:bg-slate-700 disabled:opacity-50"
              >
                {status === "saving" ? "Saving…" : "Save to MongoDB"}
              </button>
            )}
          </div>
          <AssessmentView assessment={assessment} />
        </div>
      )}
    </div>
  );
}
