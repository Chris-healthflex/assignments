import { useEffect, useState } from "react";

import { elapsed } from "../lib/format";

/**
 * The wait.
 *
 * There is no real progress to report, because the server does not stream one,
 * so this shows honest elapsed time and explains why it takes as long as it does,
 * rather than animating a bar that pretends to know. On a cold recording the
 * transcription alone runs for minutes; a spinner with no explanation looks
 * indistinguishable from a hang.
 */
const STAGES = [
  { after: 0, label: "Transcribing", note: "Whisper is running locally on the CPU." },
  {
    after: 20,
    label: "Transcribing",
    note: "First pass over a new recording takes a few minutes. The same file is cached afterwards and returns almost instantly.",
  },
  {
    after: 150,
    label: "Extracting",
    note: "The agent is reading the transcript section by section, quoting its source for every value it returns.",
  },
];

export function Progress({ filename, onCancel }) {
  const [seconds, setSeconds] = useState(0);

  useEffect(() => {
    const timer = setInterval(() => setSeconds((s) => s + 1), 1000);
    return () => clearInterval(timer);
  }, []);

  const stage = [...STAGES].reverse().find((s) => seconds >= s.after) ?? STAGES[0];

  return (
    <div className="progress">
      <h2>{stage.label}</h2>
      <p className="filename">{filename}</p>
      <div className="bar">
        <i />
      </div>
      <p className="note">{stage.note}</p>
      <p className="firstrun">
        <b>First run on this machine?</b> Whisper downloads about 1.5 GB of model
        weights before it can start, so allow roughly 3 minutes in total. The
        weights are kept, and the transcript is cached against the audio, so
        later recordings skip the download and this exact file returns at once.
      </p>
      <p className="clock">{elapsed(seconds)} elapsed</p>
      {onCancel && (
        <button type="button" className="ghost" onClick={onCancel}>
          Cancel
        </button>
      )}
    </div>
  );
}
