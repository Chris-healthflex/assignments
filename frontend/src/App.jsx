import { useCallback, useState } from "react";

import { getAssessment, parseRecording, saveAssessment } from "./lib/api";
import { Browse } from "./components/Browse";
import { Header } from "./components/Header";
import { Progress } from "./components/Progress";
import { Review } from "./components/Review";
import { Uploader } from "./components/Uploader";

/**
 * Three screens and the state between them.
 *
 * The edited assessment lives here rather than inside the review components,
 * because saving needs to send exactly what is on screen and reaching into a
 * context to find that out would be the wrong direction of travel.
 */
export default function App() {
  const [view, setView] = useState("new");
  const [phase, setPhase] = useState("idle"); // idle | working | review
  const [filename, setFilename] = useState("");
  const [result, setResult] = useState(null); // { payload, detail }
  const [assessment, setAssessment] = useState(null);
  const [readOnly, setReadOnly] = useState(false);
  const [saveState, setSaveState] = useState({ status: "idle" });
  const [banner, setBanner] = useState(null);

  const fail = useCallback((message) => {
    setBanner(message);
    setTimeout(() => setBanner(null), 9000);
  }, []);

  const start = useCallback(
    async (file) => {
      setFilename(file.name);
      setPhase("working");
      setSaveState({ status: "idle" });
      try {
        // A 422 comes back here as a result, not an exception: low confidence is
        // an answer about the recording, not a failure of the request.
        const { payload, detail } = await parseRecording(file);
        setResult({ payload, detail });
        setAssessment(payload.assessment);
        setReadOnly(false);
        setPhase("review");
      } catch (error) {
        fail(`Could not parse that recording: ${error.message}`);
        setPhase("idle");
      }
    },
    [fail],
  );

  const save = useCallback(async () => {
    setSaveState({ status: "saving" });
    try {
      const saved = await saveAssessment({
        audioFilename: result.payload.audioFilename ?? filename,
        transcript: result.payload.transcript ?? "",
        // The flags are kept as the extraction produced them, even where a value
        // has since been corrected. They are the record of what the machine
        // heard; rewriting them to match the correction would erase exactly the
        // provenance that makes the correction worth trusting.
        flags: result.payload.flags,
        assessment,
      });
      setSaveState({ status: "saved", id: saved.id });
    } catch (error) {
      fail(`Could not save: ${error.message}`);
      setSaveState({ status: "idle" });
    }
  }, [assessment, fail, filename, result]);

  const open = useCallback(
    async (id) => {
      try {
        const payload = await getAssessment(id);
        setResult({ payload, detail: [] });
        setAssessment(payload.assessment);
        setReadOnly(true);
        setPhase("review");
        setView("new");
      } catch (error) {
        fail(`Could not open that assessment: ${error.message}`);
      }
    },
    [fail],
  );

  const restart = useCallback(() => {
    setPhase("idle");
    setResult(null);
    setAssessment(null);
    setReadOnly(false);
    setSaveState({ status: "idle" });
  }, []);

  return (
    <>
      <Header view={view} onView={setView} />
      <main>
        {banner && <div className="banner">{banner}</div>}

        {view === "browse" ? (
          <Browse onOpen={open} onError={fail} />
        ) : phase === "idle" ? (
          <Uploader onStart={start} />
        ) : phase === "working" ? (
          <Progress filename={filename} />
        ) : (
          <Review
            payload={result.payload}
            detail={result.detail}
            assessment={assessment}
            onChange={setAssessment}
            readOnly={readOnly}
            onSave={save}
            saveState={saveState}
            onRestart={restart}
          />
        )}
      </main>
    </>
  );
}
