/**
 * Checks for the parts of the UI that are logic rather than markup.
 *
 * The important one is `score`. The server does not serialise the combined
 * confidence -- it is a computed property there, so that changing the rule
 * cannot leave old saved records scored by the old one -- which means the rule
 * exists in two places. These assertions mirror the Python tests in
 * tests/test_schema.py case for case, so the two copies cannot drift quietly.
 *
 *   node scripts/check-logic.mjs
 */
import { GARBLED, score } from "../src/lib/confidence.js";
import { getAt, pathFromLoc, setAt, toSteps } from "../src/lib/paths.js";
import { findSpan, splitAround } from "../src/lib/transcript.js";
import { rejectionReason, suffixOf } from "../src/lib/audio.js";

let failures = 0;

function check(name, condition) {
  if (condition) {
    console.log(`  pass  ${name}`);
  } else {
    failures += 1;
    console.log(`  FAIL  ${name}`);
  }
}

function group(title, body) {
  console.log(`\n${title}`);
  body();
}

const near = (a, b) => Math.abs(a - b) < 1e-9;

group("Paths agree with the server's dotted form", () => {
  check("an array index becomes a number step", JSON.stringify(toSteps("tests[1].left")) === '["tests",1,"left"]');
  check(
    "loc from a 422 inverts to the form field path",
    pathFromLoc(["assessment", "objectiveAssessment", "tests", 1, "left"]) ===
      "objectiveAssessment.tests[1].left",
  );
  check(
    "a flat loc drops only the assessment prefix",
    pathFromLoc(["assessment", "clinicalDetails", "duration"]) === "clinicalDetails.duration",
  );
  check(
    "a top-level array keeps its index",
    pathFromLoc(["assessment", "recommendation", 0, "sessionType"]) === "recommendation[0].sessionType",
  );
});

group("Edits are immutable, so React actually re-renders", () => {
  const before = {
    objectiveAssessment: { tests: [{ left: "1" }, { left: "20" }] },
    clinicalDetails: { duration: "x" },
  };
  const after = setAt(before, "objectiveAssessment.tests[1].left", "-5");

  check("the new value is written", getAt(after, "objectiveAssessment.tests[1].left") === "-5");
  check("the original is untouched", before.objectiveAssessment.tests[1].left === "20");
  check("the changed branch is a new object", after.objectiveAssessment !== before.objectiveAssessment);
  check("an untouched branch is shared, not cloned", after.clinicalDetails === before.clinicalDetails);
  check("siblings survive", getAt(after, "objectiveAssessment.tests[0].left") === "1");
});

group("Confidence mirrors FieldEvidence.confidence on the server", () => {
  check("no evidence means no confidence", score({ evidenceFound: false, modelConfidence: 0.9, audioConfidence: 0.9 }) === 0);
  check(
    "the weakest reported signal wins",
    near(score({ evidenceFound: true, modelConfidence: 0.95, audioConfidence: 0.52 }), 0.52),
  );
  check(
    "a signal of exactly zero counts as unreported, not as certainty",
    near(score({ evidenceFound: true, modelConfidence: 0, audioConfidence: 0.8 }), 0.8),
  );
  check(
    "a destroyed neighbouring word drags a clean quote down",
    near(
      score({ evidenceFound: true, modelConfidence: 0.95, audioConfidence: 0.99, contextConfidence: 0.05 }),
      0.05,
    ),
  );
  check(
    "an ordinary mumble nearby does not",
    near(
      score({ evidenceFound: true, modelConfidence: 0.95, audioConfidence: 0.9, contextConfidence: 0.5 }),
      0.9,
    ),
  );
  check("nothing reported at all scores zero", score({ evidenceFound: true }) === 0);
  check("the garbled threshold matches the server's", GARBLED === 0.25);
});

group("Quotes survive punctuation drift", () => {
  const transcript = "left knee flexion of 124 degrees, with 130 degrees on the right";
  check("an exact quote is found", findSpan(transcript, "124 degrees") !== null);
  check("punctuation between words is tolerated", findSpan(transcript, "124 degrees with") !== null);
  check("a quote that is not there is not invented", findSpan(transcript, "hip abduction 40") === null);

  const parts = splitAround(transcript, "130 degrees");
  check("the highlight lands on the right words", parts !== null && parts[1] === "130 degrees");
  check("the surrounding text is preserved", parts !== null && parts.join("") === transcript);
});

group("Uploads are screened before fifty megabytes go over the wire", () => {
  const file = (name, size) => ({ name, size });
  check("suffix is read case-insensitively", suffixOf("Session.WAV") === ".wav");
  check("wav is accepted", rejectionReason(file("a.wav", 1000)) === null);
  check("mp3 is accepted", rejectionReason(file("a.mp3", 1000)) === null);
  check("m4a is accepted", rejectionReason(file("a.m4a", 1000)) === null);
  check("pdf is refused", rejectionReason(file("a.pdf", 1000)) !== null);
  check("an empty file is refused", rejectionReason(file("a.wav", 0)) !== null);
  check("an oversized file is refused", rejectionReason(file("a.wav", 60 * 1024 * 1024)) !== null);
});

console.log(failures ? `\n${failures} check(s) failed` : "\nall checks passed");
process.exit(failures ? 1 : 0);
