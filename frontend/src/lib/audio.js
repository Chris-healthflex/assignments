/**
 * What the service accepts.
 *
 * These mirror `AUDIO_SUFFIXES` in app/main.py. faster-whisper decodes through
 * the PyAV bindings it ships with, so mp3, m4a, flac and ogg need no ffmpeg
 * install of their own -- all three of wav, mp3 and m4a were verified to
 * transcribe to identical text from the same source recording.
 *
 * Checking here as well as on the server is not duplication for its own sake:
 * rejecting a file locally saves uploading fifty megabytes to learn it was the
 * wrong kind. The server check is the one that counts.
 */
export const ACCEPTED_SUFFIXES = [".wav", ".wave", ".mp3", ".m4a", ".flac", ".ogg", ".webm"];

/** Value for an <input type="file"> accept attribute. */
export const ACCEPT_ATTRIBUTE = [
  "audio/wav",
  "audio/x-wav",
  "audio/mpeg",
  "audio/mp4",
  "audio/x-m4a",
  "audio/flac",
  "audio/ogg",
  "audio/webm",
  ...ACCEPTED_SUFFIXES,
].join(",");

export const MAX_UPLOAD_MB = 50;

export function suffixOf(filename) {
  const dot = filename.lastIndexOf(".");
  return dot < 0 ? "" : filename.slice(dot).toLowerCase();
}

/** `null` when the file is fine, otherwise the reason it is not. */
export function rejectionReason(file) {
  const suffix = suffixOf(file.name);
  if (!ACCEPTED_SUFFIXES.includes(suffix)) {
    return `${suffix || "That file"} is not an audio format this service reads. Use ${ACCEPTED_SUFFIXES.slice(0, 3).join(", ")} or similar.`;
  }
  if (file.size === 0) return "That file is empty.";
  if (file.size > MAX_UPLOAD_MB * 1024 * 1024) {
    return `That file is ${(file.size / 1048576).toFixed(0)} MB; the limit is ${MAX_UPLOAD_MB} MB.`;
  }
  return null;
}

export function humanSize(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1048576) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / 1048576).toFixed(1)} MB`;
}
