import { useEffect, useRef } from "react";

import { GARBLED, pct, severityOf } from "../lib/confidence";
import { splitAround } from "../lib/transcript";
import { useAssessment } from "../state/AssessmentContext";

/**
 * What the recording actually said.
 *
 * Selecting any field scrolls the transcript to the words it came from and
 * shows the three signals behind its score. This is the part of the design the
 * whole pipeline exists to make possible: a clinician can check a number
 * against the sentence it came from without leaving the screen, and without
 * taking anyone's word for where it came from.
 */
export function Inspector({ transcript }) {
  const { selected, flagFor, evidenceFor, valueAt, confidenceAt } = useAssessment();
  const markRef = useRef(null);

  const flag = selected ? flagFor(selected) : null;
  const evidence = selected ? evidenceFor(selected) : null;
  const quote = flag?.evidence || evidence?.evidence || "";
  const parts = quote ? splitAround(transcript, quote) : null;

  useEffect(() => {
    markRef.current?.scrollIntoView({ block: "center", behavior: "smooth" });
  }, [selected, quote]);

  return (
    <aside className="inspector">
      <div className="panel">
        <header className="panel-head">
          <h3>Transcript</h3>
          {quote && !parts && <span className="tag bad">quote not located</span>}
        </header>
        <div className="transcript">
          {parts ? (
            <>
              {parts[0]}
              <mark ref={markRef}>{parts[1]}</mark>
              {parts[2]}
            </>
          ) : (
            transcript || "No transcript."
          )}
        </div>
      </div>

      <div className="panel">
        <header className="panel-head">
          <h3>Evidence</h3>
        </header>
        {selected ? (
          <FieldDetail
            path={selected}
            flag={flag}
            evidence={evidence}
            confidence={confidenceAt(selected)}
            current={valueAt(selected)}
          />
        ) : (
          <p className="empty">Select any field to see what was heard and how clearly.</p>
        )}
      </div>
    </aside>
  );
}

function FieldDetail({ path, flag, evidence, confidence, current }) {
  const severity = severityOf(flag);
  const source = flag ?? evidence;

  if (!source) {
    return (
      <div className="detail">
        <Row label="Field" value={path} mono />
        <p className="empty">
          Nothing was extracted for this field, so there is no evidence to show. An empty
          field means the recording did not state it.
        </p>
      </div>
    );
  }

  const original = flag?.value ?? evidence?.value;
  const edited = original != null && original !== current;

  return (
    <div className="detail">
      <Row label="Field" value={path} mono />
      <Row
        label="Quoted from the recording"
        value={source.evidence ? `“${source.evidence}”` : "— nothing quoted —"}
        mono
      />

      {flag && <p className={`why ${severity}`}>{flag.msg}</p>}
      {edited && (
        <p className="edited">
          Extraction had <b>“{original}”</b> — you changed it to <b>“{current}”</b>.
        </p>
      )}

      <div className="signals">
        <Signal label="Model self-report" value={source.modelConfidence} />
        <Signal label="Whisper on these words" value={source.audioConfidence} />
        <Signal label="Whisper on the words around them" value={source.contextConfidence} />
      </div>

      <div className="combined">
        <span>Combined</span>
        <strong className={severity ?? ""}>{pct(confidence)}</strong>
      </div>
    </div>
  );
}

function Row({ label, value, mono }) {
  return (
    <div className="kv">
      <span className="k">{label}</span>
      <span className={mono ? "v mono" : "v"}>{value}</span>
    </div>
  );
}

/**
 * A signal that was never measured shows as "not measured" rather than 0%.
 * Absent and zero mean opposite things here: one is "no information", the other
 * is "measured, and bad".
 */
function Signal({ label, value }) {
  if (value == null) {
    return (
      <div className="signal">
        <span className="k">{label}</span>
        <span className="v muted">not measured</span>
      </div>
    );
  }
  const tone = value < GARBLED ? "bad" : value < 0.6 ? "warn" : "";
  return (
    <div className="signal">
      <span className="k">{label}</span>
      <span className="v">{pct(value)}</span>
      <div className={`meter ${tone}`}>
        <i style={{ width: `${Math.max(2, value * 100)}%` }} />
      </div>
    </div>
  );
}
