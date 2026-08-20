import { useRef, useState } from "react";

import { ACCEPT_ATTRIBUTE, MAX_UPLOAD_MB, humanSize, rejectionReason } from "../lib/audio";

/**
 * Choosing a recording.
 *
 * The format and size checks run here as well as on the server. Not because the
 * client's opinion counts, since the server rejects the same files
 * independently, but because finding out a file is the wrong kind after
 * uploading forty megabytes over a home connection is a bad way to learn it.
 */
export function Uploader({ onStart, disabled }) {
  const inputRef = useRef(null);
  const [file, setFile] = useState(null);
  const [problem, setProblem] = useState(null);
  const [dragging, setDragging] = useState(false);

  function choose(candidate) {
    if (!candidate) return;
    const reason = rejectionReason(candidate);
    setProblem(reason);
    setFile(reason ? null : candidate);
  }

  return (
    <div
      className={`dropzone ${dragging ? "is-over" : ""}`}
      onDragEnter={(event) => {
        event.preventDefault();
        setDragging(true);
      }}
      onDragOver={(event) => event.preventDefault()}
      onDragLeave={() => setDragging(false)}
      onDrop={(event) => {
        event.preventDefault();
        setDragging(false);
        choose(event.dataTransfer.files?.[0]);
      }}
    >
      <h2>Upload a consultation recording</h2>
      <p className="lede">
        WAV, MP3 or M4A, plus FLAC, OGG and WebM. Up to {MAX_UPLOAD_MB} MB.
        <br />
        Nothing is saved until you have reviewed it.
      </p>

      <input
        ref={inputRef}
        type="file"
        accept={ACCEPT_ATTRIBUTE}
        hidden
        onChange={(event) => choose(event.target.files?.[0])}
      />

      <div className="actions">
        <button type="button" className="ghost" onClick={() => inputRef.current?.click()}>
          Choose file
        </button>
        <button
          type="button"
          className="primary"
          disabled={!file || disabled}
          onClick={() => file && onStart(file)}
        >
          Transcribe &amp; extract
        </button>
      </div>

      {problem ? (
        <p className="problem">{problem}</p>
      ) : file ? (
        <p className="chosen">
          <b>{file.name}</b> · {humanSize(file.size)}
        </p>
      ) : (
        <p className="chosen">or drop a file here</p>
      )}
    </div>
  );
}
