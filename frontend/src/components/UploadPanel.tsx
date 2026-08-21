import { useRef, useState } from "react";
import { ApiError, parseAssessment, saveAssessment } from "../api";
import type { FirstAssessment, ParseDebugResult } from "../types";
import { AssessmentView } from "./AssessmentView";
import { ConfidenceBadge } from "./ConfidenceBadge";

type Status = "idle" | "parsing" | "parsed" | "saving" | "saved" | "error";

export function UploadPanel() {
  const [status, setStatus] = useState<Status>("idle");
  const [fileName, setFileName] = useState<string>("");
  const [parseResult, setParseResult] = useState<ParseDebugResult | null>(null);
  const [assessment, setAssessment] = useState<FirstAssessment | null>(null);
  const [error, setError] = useState<string>("");
  const [showTranscript, setShowTranscript] = useState(false);
  const [savedId, setSavedId] = useState<string>("");
  const [isDragging, setIsDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const isSavingRef = useRef(false);

  async function handleFile(file: File) {
    setFileName(file.name);
    setStatus("parsing");
    setError("");
    setParseResult(null);
    setAssessment(null);
    setSavedId("");
    isSavingRef.current = false;

    try {
      const result = await parseAssessment(file);
      setParseResult(result);
      setAssessment(result.assessment);
      setStatus("parsed");
    } catch (err) {
      if (err instanceof ApiError) {
        setError(typeof err.detail === "string" ? err.detail : "Failed to parse audio");
      } else {
        setError("Could not reach the API. Is the FastAPI server running?");
      }
      setStatus("error");
    }
  }

  async function handleSave() {
    if (!assessment || isSavingRef.current) return;
    isSavingRef.current = true;
    setStatus("saving");
    try {
      const { id } = await saveAssessment(assessment);
      setSavedId(id);
      setStatus("saved");
    } catch {
      isSavingRef.current = false;
      setError("Failed to save assessment");
      setStatus("error");
    }
  }

  return (
    <div className="space-y-6">
      {status === "idle" && (
        <div>
          <h2 className="text-2xl font-semibold text-slate-900">
            Turn a session recording into a structured assessment
          </h2>
          <p className="mt-1 max-w-xl text-sm text-slate-600">
            Upload a clinician–patient audio session. We'll transcribe it and
            extract clinical details, measurements, goals, and recommendations
            into a structured record — flagging anything the recording didn't
            cover so you can complete it by hand.
          </p>
        </div>
      )}

      <div
        className={`flex flex-col items-center justify-center rounded-2xl border-2 border-dashed p-10 text-center transition ${
          isDragging
            ? "border-teal-500 bg-teal-50"
            : "border-slate-300 bg-white hover:border-teal-400"
        }`}
        onDragOver={(e) => {
          e.preventDefault();
          setIsDragging(true);
        }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setIsDragging(false);
          const file = e.dataTransfer.files?.[0];
          if (file) handleFile(file);
        }}
      >
        <WaveformIcon className="mb-3 h-8 w-8 text-teal-500" />
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
        {fileName ? (
          <p className="mt-4 text-xs text-slate-500">{fileName}</p>
        ) : (
          <p className="mt-4 text-xs text-slate-400">WAV audio only</p>
        )}
      </div>

      {status === "idle" && <HowItWorks />}

      {status === "parsing" && (
        <div className="space-y-2 rounded-xl border border-slate-200 bg-white p-5">
          <ProcessingSteps />
        </div>
      )}

      {status === "error" && (
        <div className="rounded-xl border border-amber-300 bg-amber-50 p-4 text-sm text-amber-900">
          <p className="font-medium">{error || "Something went wrong"}</p>
        </div>
      )}

      {parseResult && assessment && (
        <div className="space-y-4">
          <div className="flex items-start justify-between gap-4">
            <h2 className="text-lg font-semibold text-slate-900">Extracted Assessment</h2>
            {status === "saved" ? (
              <span className="rounded-full bg-teal-100 px-3 py-1 text-xs font-medium text-teal-800">
                Saved · id {savedId}
              </span>
            ) : (
              <button
                onClick={handleSave}
                disabled={status === "saving"}
                className="shrink-0 rounded-lg bg-slate-900 px-4 py-2 text-sm font-medium text-white shadow-sm transition hover:bg-slate-700 disabled:opacity-50"
              >
                {status === "saving" ? "Saving…" : "Save to MongoDB"}
              </button>
            )}
          </div>

          <div className="grid gap-4 sm:grid-cols-[minmax(0,1fr)_auto]">
            <ConfidenceBadge
              confidence={parseResult.confidence}
              flaggedCount={parseResult.low_confidence_sections.length}
            />
            <button
              onClick={() => setShowTranscript((v) => !v)}
              className="rounded-xl border border-slate-200 bg-white px-4 py-2 text-sm font-medium text-slate-700 shadow-sm transition hover:border-teal-300 sm:self-start"
            >
              {showTranscript ? "Hide transcript" : "Show transcript"}
            </button>
          </div>

          {showTranscript && (
            <div className="max-h-64 overflow-y-auto rounded-xl border border-slate-200 bg-slate-900 p-4 text-sm leading-relaxed text-slate-100">
              {parseResult.transcript}
            </div>
          )}

          {parseResult.low_confidence_sections.length > 0 && (
            <p className="text-sm text-slate-600">
              Sections highlighted below weren't covered in the recording — click
              into them to complete by hand before saving.
            </p>
          )}

          <AssessmentView
            assessment={assessment}
            flaggedSections={parseResult.low_confidence_sections}
            editable
            onChange={setAssessment}
          />
        </div>
      )}
    </div>
  );
}

function WaveformIcon({ className }: { className?: string }) {
  const bars = [6, 12, 18, 10, 16, 8, 14, 6];
  return (
    <svg viewBox="0 0 64 24" className={className} fill="none">
      {bars.map((h, i) => (
        <rect
          key={i}
          x={i * 8}
          y={(24 - h) / 2}
          width={4}
          height={h}
          rx={2}
          fill="currentColor"
        />
      ))}
    </svg>
  );
}

function HowItWorks() {
  const steps = [
    {
      title: "1. Upload",
      body: "Drop in a WAV recording of a clinician–patient session.",
    },
    {
      title: "2. Transcribe & extract",
      body: "Whisper transcribes the audio, then a LangGraph pipeline pulls out clinical details, measurements, goals, and recommendations.",
    },
    {
      title: "3. Review & save",
      body: "Flagged sections the recording didn't cover are highlighted — fill them in, then save to MongoDB.",
    },
  ];

  return (
    <div className="grid gap-4 border-t border-slate-200 pt-6 sm:grid-cols-3">
      {steps.map((step) => (
        <div key={step.title}>
          <p className="text-sm font-semibold text-slate-900">{step.title}</p>
          <p className="mt-1 text-sm text-slate-500">{step.body}</p>
        </div>
      ))}
    </div>
  );
}

function ProcessingSteps() {
  const steps = ["Transcribing audio", "Extracting clinical data", "Checking confidence"];
  return (
    <ul className="space-y-2">
      {steps.map((step, i) => (
        <li key={step} className="flex items-center gap-2 text-sm text-slate-600">
          <span
            className="h-2 w-2 shrink-0 animate-pulse rounded-full bg-teal-500"
            style={{ animationDelay: `${i * 0.2}s` }}
          />
          {step}
        </li>
      ))}
      <p className="pt-1 text-xs text-slate-400">
        Running Whisper transcription and the LangGraph extraction pipeline —
        this can take a little while on a long recording.
      </p>
    </ul>
  );
}
