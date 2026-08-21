import { useEffect, useRef, useState } from "react";
import type { TranscriptSegment } from "../types";
import { Card } from "./ui/Card";

interface Props {
  segments: TranscriptSegment[];
  audioUrl: string | null;
  /** Segments cited as evidence for the field the reviewer is inspecting. */
  citedSegmentIds: number[];
  citedFieldLabel: string | null;
  onClearCitation: () => void;
}

export function formatTime(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds < 0) return "0:00";
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins}:${secs.toString().padStart(2, "0")}`;
}

/**
 * The verification surface: the recording, its time-coded transcript, and the
 * link between the two.
 *
 * The pipeline's hardest promise is that it never invents clinical values. This
 * is where a clinician checks that promise — select a field and the exact
 * segments the model cited light up and the audio jumps there, so a wrong value
 * takes seconds to catch instead of a full re-listen.
 */
export function EvidencePanel({
  segments,
  audioUrl,
  citedSegmentIds,
  citedFieldLabel,
  onClearCitation,
}: Props) {
  const audioRef = useRef<HTMLAudioElement>(null);
  const listRef = useRef<HTMLDivElement>(null);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);

  const cited = new Set(citedSegmentIds);

  // Follow along with playback so the reviewer can see where they are.
  useEffect(() => {
    const audio = audioRef.current;
    if (!audio) return;

    const onTime = () => setCurrentTime(audio.currentTime);
    const onMeta = () => setDuration(audio.duration);
    const onPlay = () => setIsPlaying(true);
    const onPause = () => setIsPlaying(false);

    audio.addEventListener("timeupdate", onTime);
    audio.addEventListener("loadedmetadata", onMeta);
    audio.addEventListener("play", onPlay);
    audio.addEventListener("pause", onPause);
    audio.addEventListener("ended", onPause);

    return () => {
      audio.removeEventListener("timeupdate", onTime);
      audio.removeEventListener("loadedmetadata", onMeta);
      audio.removeEventListener("play", onPlay);
      audio.removeEventListener("pause", onPause);
      audio.removeEventListener("ended", onPause);
    };
  }, [audioUrl]);

  // When a field is selected, jump to its first citation and scroll it in view.
  useEffect(() => {
    if (citedSegmentIds.length === 0) return;

    const first = segments.find((segment) => segment.id === citedSegmentIds[0]);
    if (!first) return;

    seek(first.start);
    const row = listRef.current?.querySelector(`[data-segment-id="${first.id}"]`);
    // Scrolling is a nicety, and not every environment implements it.
    row?.scrollIntoView?.({ block: "center", behavior: "smooth" });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [citedSegmentIds.join(",")]);

  function seek(seconds: number) {
    const audio = audioRef.current;
    if (audio) audio.currentTime = seconds;
    setCurrentTime(seconds);
  }

  function togglePlay() {
    const audio = audioRef.current;
    if (!audio) return;
    if (audio.paused) void audio.play();
    else audio.pause();
  }

  const activeSegment = segments.find(
    (segment) => currentTime >= segment.start && currentTime < segment.end,
  );

  return (
    <Card className="overflow-hidden">
      {audioUrl && (
        <div className="flex items-center gap-3 border-b border-slate-200 bg-slate-50 px-4 py-3 dark:border-slate-800 dark:bg-slate-900">
          <audio ref={audioRef} src={audioUrl} preload="metadata" />
          <button
            onClick={togglePlay}
            aria-label={isPlaying ? "Pause recording" : "Play recording"}
            className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full bg-teal-600 text-white transition hover:bg-teal-700"
          >
            {isPlaying ? <PauseIcon /> : <PlayIcon />}
          </button>

          <input
            type="range"
            min={0}
            max={duration || 0}
            step={0.1}
            value={currentTime}
            aria-label="Seek recording"
            onChange={(e) => seek(Number(e.target.value))}
            className="h-1 flex-1 cursor-pointer appearance-none rounded-full bg-slate-300 accent-teal-600 dark:bg-slate-700"
          />

          <span className="shrink-0 font-mono text-xs tabular-nums text-slate-500 dark:text-slate-400">
            {formatTime(currentTime)} / {formatTime(duration)}
          </span>
        </div>
      )}

      {citedFieldLabel && (
        <div className="flex items-center justify-between gap-3 border-b border-teal-200 bg-teal-50 px-4 py-2 text-xs dark:border-teal-900 dark:bg-teal-950/50">
          <p className="text-teal-800 dark:text-teal-300">
            Showing the evidence for <strong>{citedFieldLabel}</strong>
            {citedSegmentIds.length === 0 && " — the model cited no segment"}
          </p>
          <button
            onClick={onClearCitation}
            className="shrink-0 font-medium text-teal-700 hover:underline dark:text-teal-400"
          >
            Clear
          </button>
        </div>
      )}

      <div ref={listRef} className="max-h-80 space-y-1 overflow-y-auto p-3">
        {segments.map((segment) => {
          const isCited = cited.has(segment.id);
          const isActive = activeSegment?.id === segment.id;

          return (
            <button
              key={segment.id}
              data-segment-id={segment.id}
              onClick={() => seek(segment.start)}
              className={`flex w-full gap-3 rounded-lg px-3 py-2 text-left text-sm transition ${
                isCited
                  ? "bg-teal-100 text-teal-950 ring-1 ring-teal-400 dark:bg-teal-900/50 dark:text-teal-100 dark:ring-teal-700"
                  : isActive
                    ? "bg-slate-100 text-slate-900 dark:bg-slate-800 dark:text-slate-100"
                    : "text-slate-600 hover:bg-slate-50 dark:text-slate-400 dark:hover:bg-slate-900"
              }`}
            >
              <span className="shrink-0 pt-0.5 font-mono text-xs tabular-nums text-slate-400">
                {formatTime(segment.start)}
              </span>
              <span className="leading-relaxed">{segment.text}</span>
            </button>
          );
        })}
      </div>
    </Card>
  );
}

function PlayIcon() {
  return (
    <svg viewBox="0 0 24 24" className="h-4 w-4 translate-x-px" fill="currentColor">
      <path d="M8 5v14l11-7z" />
    </svg>
  );
}

function PauseIcon() {
  return (
    <svg viewBox="0 0 24 24" className="h-4 w-4" fill="currentColor">
      <path d="M6 5h4v14H6zM14 5h4v14h-4z" />
    </svg>
  );
}
