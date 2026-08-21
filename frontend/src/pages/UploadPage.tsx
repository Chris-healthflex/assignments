import { useEffect, useMemo, useRef, useState } from "react";
import { useParseAssessment } from "../hooks/useParseAssessment";
import { useSaveAssessment } from "../hooks/useSaveAssessment";
import type { FirstAssessment } from "../types";
import { AssessmentView } from "../components/AssessmentView";
import { ConfidenceBadge } from "../components/ConfidenceBadge";
import { EvidencePanel } from "../components/EvidencePanel";
import { useRegisterCommands } from "../components/CommandPalette";
import { Button } from "../components/ui/Button";
import { Card } from "../components/ui/Card";
import { Badge } from "../components/ui/Badge";
import { useToast } from "../components/ui/Toast";

interface InspectedField {
  path: string;
  label: string;
  segmentIds: number[];
}

export function UploadPage() {
  const {
    status: parseStatus,
    result: parseResult,
    error: parseError,
    parse,
    reset: resetParse,
  } = useParseAssessment();
  const { status: saveStatus, savedId, save, reset: resetSave } = useSaveAssessment();

  const [assessment, setAssessment] = useState<FirstAssessment | null>(null);
  const [fileName, setFileName] = useState<string>("");
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [inspected, setInspected] = useState<InspectedField | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const toast = useToast();

  useEffect(() => {
    if (parseResult) setAssessment(parseResult.assessment);
  }, [parseResult]);

  // The recording never leaves the browser for playback — we hand the <audio>
  // element a blob URL for the very file that was uploaded, so evidence review
  // needs no extra round trip and no server-side audio storage.
  useEffect(() => () => {
    if (audioUrl) URL.revokeObjectURL(audioUrl);
  }, [audioUrl]);

  const hasUnsavedWork = parseResult !== null && saveStatus !== "saved";

  useEffect(() => {
    if (!hasUnsavedWork) return;
    const handler = (e: BeforeUnloadEvent) => {
      e.preventDefault();
    };
    window.addEventListener("beforeunload", handler);
    return () => window.removeEventListener("beforeunload", handler);
  }, [hasUnsavedWork]);

  useEffect(() => {
    if (saveStatus === "saved") toast.show("Assessment saved to MongoDB");
    if (saveStatus === "error") toast.show("Failed to save assessment", "error");
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [saveStatus]);

  async function handleFile(file: File) {
    setFileName(file.name);
    setAudioUrl((previous) => {
      if (previous) URL.revokeObjectURL(previous);
      return URL.createObjectURL(file);
    });
    setInspected(null);
    resetSave();
    await parse(file);
  }

  async function handleSave() {
    if (!assessment) return;
    await save(assessment);
  }

  function downloadJson() {
    if (!assessment) return;
    const blob = new Blob([JSON.stringify(assessment, null, 2)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `${fileName.replace(/\.wav$/i, "") || "assessment"}.json`;
    anchor.click();
    URL.revokeObjectURL(url);
  }

  useRegisterCommands(
    assessment
      ? [
          {
            id: "save-assessment",
            group: "Assessment",
            label: "Save to MongoDB",
            run: () => void handleSave(),
          },
          {
            id: "export-json",
            group: "Assessment",
            label: "Download as JSON",
            run: downloadJson,
          },
          {
            id: "choose-file",
            group: "Assessment",
            label: "Upload a different recording",
            run: () => inputRef.current?.click(),
          },
        ]
      : [
          {
            id: "choose-file",
            group: "Assessment",
            label: "Upload a recording",
            run: () => inputRef.current?.click(),
          },
        ],
    [assessment, fileName, saveStatus],
  );

  const ungroundedCount = parseResult?.ungrounded_fields.length ?? 0;
  const segments = useMemo(() => parseResult?.segments ?? [], [parseResult]);

  return (
    <div className="space-y-6">
      {parseStatus === "idle" && (
        <div>
          <h2 className="text-2xl font-semibold text-slate-900 dark:text-slate-100">
            Turn a session recording into a structured assessment
          </h2>
          <p className="mt-1 max-w-xl text-sm text-slate-600 dark:text-slate-400">
            Upload a clinician–patient audio session. We'll transcribe it and
            extract clinical details, measurements, goals, and recommendations
            into a structured record — citing the moment in the recording each
            value came from, and flagging anything it couldn't find.
          </p>
        </div>
      )}

      <div
        className={`flex flex-col items-center justify-center rounded-2xl border-2 border-dashed p-10 text-center transition ${
          isDragging
            ? "border-teal-500 bg-teal-50 dark:bg-teal-950/40"
            : "border-slate-300 bg-white hover:border-teal-400 dark:border-slate-700 dark:bg-slate-900"
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
        <p className="mb-1 text-sm font-medium text-slate-700 dark:text-slate-300">
          Drop a clinical session WAV file here
        </p>
        <p className="mb-4 text-xs text-slate-500">or</p>
        <Button onClick={() => inputRef.current?.click()}>Choose file</Button>
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

      {parseStatus === "idle" && <HowItWorks />}

      {parseStatus === "parsing" && (
        <Card className="p-5">
          <ProcessingSteps />
        </Card>
      )}

      {parseStatus === "error" && (
        <div className="rounded-xl border border-amber-300 bg-amber-50 p-4 text-sm text-amber-900 dark:border-amber-800 dark:bg-amber-950/40 dark:text-amber-200">
          <div className="flex items-start justify-between gap-4">
            <p className="font-medium">{parseError || "Something went wrong"}</p>
            <Button variant="ghost" onClick={resetParse} className="shrink-0">
              Dismiss
            </Button>
          </div>
        </div>
      )}

      {parseResult && assessment && (
        <div className="space-y-4">
          <div className="flex items-start justify-between gap-4">
            <h2 className="text-lg font-semibold text-slate-900 dark:text-slate-100">
              Extracted Assessment
            </h2>
            <div className="flex shrink-0 items-center gap-2">
              <Button variant="secondary" onClick={downloadJson}>
                Download JSON
              </Button>
              {saveStatus === "saved" ? (
                <Badge tone="teal">Saved · id {savedId}</Badge>
              ) : (
                <Button
                  variant="dark"
                  onClick={handleSave}
                  disabled={saveStatus === "saving"}
                >
                  {saveStatus === "saving" ? "Saving…" : "Save to MongoDB"}
                </Button>
              )}
            </div>
          </div>

          <ConfidenceBadge
            confidence={parseResult.confidence}
            flaggedCount={parseResult.low_confidence_sections.length}
          />

          {ungroundedCount > 0 && (
            <div className="rounded-xl border border-amber-300 bg-amber-50 p-4 text-sm dark:border-amber-800 dark:bg-amber-950/40">
              <p className="font-medium text-amber-900 dark:text-amber-200">
                {ungroundedCount} field{ungroundedCount === 1 ? "" : "s"} could not
                be traced back to the recording
              </p>
              <p className="mt-1 text-amber-800 dark:text-amber-300">
                They're marked <strong>⚠ unverified</strong> below. The extraction
                agent filled them in but couldn't cite a transcript segment, so
                check them against the audio before saving.
              </p>
            </div>
          )}

          {parseResult.attempts > 1 && (
            <p className="text-xs text-slate-500 dark:text-slate-400">
              The extraction agent corrected itself {parseResult.attempts - 1} time
              {parseResult.attempts - 1 === 1 ? "" : "s"} before settling on this
              result.
            </p>
          )}

          {segments.length > 0 && (
            <EvidencePanel
              segments={segments}
              audioUrl={audioUrl}
              citedSegmentIds={inspected?.segmentIds ?? []}
              citedFieldLabel={inspected?.label ?? null}
              onClearCitation={() => setInspected(null)}
            />
          )}

          {parseResult.low_confidence_sections.length > 0 && (
            <p className="text-sm text-slate-600 dark:text-slate-400">
              Sections highlighted below weren't covered in the recording — click
              into them to complete by hand before saving.
            </p>
          )}

          <AssessmentView
            assessment={assessment}
            flaggedSections={parseResult.low_confidence_sections}
            evidence={parseResult.evidence}
            ungroundedFields={parseResult.ungrounded_fields}
            inspectedField={inspected?.path ?? null}
            onInspectField={(path, label, segmentIds) =>
              setInspected((current) =>
                current?.path === path ? null : { path, label, segmentIds },
              )
            }
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
      body: "Whisper produces a time-coded transcript, then a LangGraph agent extracts the assessment and cites the segment behind every value.",
    },
    {
      title: "3. Verify & save",
      body: "Click any value to hear where it came from. Anything the agent couldn't cite is flagged for review before saving.",
    },
  ];

  return (
    <div className="grid gap-4 border-t border-slate-200 pt-6 sm:grid-cols-3 dark:border-slate-800">
      {steps.map((step) => (
        <div key={step.title}>
          <p className="text-sm font-semibold text-slate-900 dark:text-slate-100">
            {step.title}
          </p>
          <p className="mt-1 text-sm text-slate-500 dark:text-slate-400">{step.body}</p>
        </div>
      ))}
    </div>
  );
}

function ProcessingSteps() {
  const steps = [
    "Transcribing audio",
    "Extracting clinical data",
    "Checking every value against the transcript",
  ];
  return (
    <ul className="space-y-2">
      {steps.map((step, i) => (
        <li
          key={step}
          className="flex items-center gap-2 text-sm text-slate-600 dark:text-slate-400"
        >
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
