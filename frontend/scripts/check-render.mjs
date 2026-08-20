/**
 * Renders the review screen to a string and checks what came out.
 *
 * `vite build` proves every import resolves and every file parses. It does not
 * prove a component survives being given a real payload. An assessment with
 * seven empty sections and a flag pointing into an array that has no rows is
 * exactly the shape that throws on the first render, and that shape is a normal
 * response from this API, not an edge case.
 *
 *   node scripts/check-render.mjs
 */
import { build } from "esbuild";
import { mkdir, rm } from "node:fs/promises";
import { fileURLToPath, pathToFileURL } from "node:url";
import { dirname, join } from "node:path";

const here = dirname(fileURLToPath(import.meta.url));
const outfile = join(here, "..", ".vite", "render-smoke.mjs");

await mkdir(dirname(outfile), { recursive: true });
await build({
  entryPoints: [join(here, "render-entry.jsx")],
  outfile,
  bundle: true,
  format: "esm",
  platform: "node",
  jsx: "automatic",
  logLevel: "error",
  // Left to resolve from node_modules at import time rather than bundled in.
  external: ["react", "react/jsx-runtime", "react-dom", "react-dom/server"],
});

const entry = await import(pathToFileURL(outfile).href);

let failures = 0;
const check = (name, condition) => {
  console.log(`  ${condition ? "pass" : "FAIL"}  ${name}`);
  if (!condition) failures += 1;
};

/* ------------------------------------------------------------------ *
 * A populated assessment with two flagged fields
 * ------------------------------------------------------------------ */
console.log("\nA parsed assessment renders");
const assessment = entry.populatedAssessment();
const html = entry.render({
  assessment,
  payload: entry.payloadFor(assessment),
  detail: entry.DETAIL,
});

const SECTIONS = [
  "Clinical details",
  "Subjective assessments",
  "Objective assessment",
  "Subjective goals",
  "Objective goals",
  "Recommendation",
  "Patient advice",
];
check("all seven contract sections appear", SECTIONS.every((title) => html.includes(title)));
check("in contract order", SECTIONS.reduce((at, title) => (at < 0 ? at : html.indexOf(title, at)), 0) > 0);
check("measurements render as a table", html.includes("<table>") && html.includes(">Left</th>"));
check("left and right are adjacent columns", html.indexOf(">Left</th>") < html.indexOf(">Right</th>"));
check("values reach their inputs", html.includes('value="124"') && html.includes('value="130"'));
check("the transcript is on the page", html.includes("left knee flexion of 124 degrees"));

console.log("\nFlagged fields are marked");
check("the worklist is present", html.includes("Needs review"));
check("a poorly heard field is amber", html.includes("flag-warn"));
check("an unsourced field is red", html.includes("flag-bad"));
check("the score is shown", html.includes("52%"));
check("the tally counts both kinds", html.includes("heard poorly") && html.includes("without a source"));
check("flagged inputs are marked invalid for screen readers", html.includes('aria-invalid="true"'));

/* ------------------------------------------------------------------ *
 * The empty case, which is a normal answer here rather than an error
 * ------------------------------------------------------------------ */
console.log("\nAn assessment where the recording said nothing renders");
const bare = entry.emptyAssessment();
const emptyHtml = entry.render({
  assessment: bare,
  payload: { ...entry.payloadFor(bare), flags: { overallConfidence: 0, fields: [] } },
  detail: [],
});
check("every section still appears", SECTIONS.every((title) => emptyHtml.includes(title)));
check("empty arrays explain themselves", emptyHtml.includes("Nothing stated in the recording."));
check("no measurement table is drawn", !emptyHtml.includes("<table>"));
check("nothing is flagged", emptyHtml.includes("nothing flagged"));
check("the worklist is absent", !emptyHtml.includes("Needs review"));

/* ------------------------------------------------------------------ *
 * A flag pointing at a row that is not there: the shape most likely
 * to throw, and reachable if a saved record is ever re-flagged.
 * ------------------------------------------------------------------ */
console.log("\nA flag with no matching row does not bring the page down");
let survived = true;
try {
  entry.render({
    assessment: bare,
    payload: { ...entry.payloadFor(bare), flags: { overallConfidence: 0, fields: [] } },
    detail: entry.DETAIL,
  });
} catch (error) {
  survived = false;
  console.log(`        ${error.message}`);
}
check("it renders instead of throwing", survived);

/* ------------------------------------------------------------------ *
 * A section lost to a failed call. The whole point is that this must
 * NOT look like a section the clinician simply did not mention.
 * ------------------------------------------------------------------ */
console.log("\nA section lost to a failed call says so");
const lost = entry.emptyAssessment();
lost.objectiveAssessment.tests = entry.populatedAssessment().objectiveAssessment.tests;
const lostPayload = entry.payloadFor(lost);
lostPayload.flags = {
  overallConfidence: 0.94,
  fields: [],
  failedSections: ["clinicalDetails"],
};
const lostHtml = entry.render({
  assessment: lost,
  payload: lostPayload,
  detail: entry.UNAVAILABLE_DETAIL,
});

check("the header warns the assessment is incomplete", lostHtml.includes("This assessment is incomplete"));
check("it names the section", lostHtml.includes("clinicalDetails"));
check("the section is marked not extracted", lostHtml.includes("not extracted"));
check(
  "it does not claim the recording was silent",
  lostHtml.includes("not</b> because the recording was silent") ||
    lostHtml.includes("not because the recording was silent"),
);
check(
  "and does not also show the ordinary empty message there",
  !lostHtml.includes("Nothing stated in the recording.") ||
    lostHtml.indexOf("could not be extracted") < lostHtml.indexOf("Nothing stated"),
);
check("a high score does not suppress the warning", lostHtml.includes("94%"));
check("the tally counts the missing section", lostHtml.includes("section(s) missing"));
check("it is not reported as nothing flagged", !lostHtml.includes("nothing flagged"));
check("the worklist lists it", lostHtml.includes("whole section"));

/* ------------------------------------------------------------------ *
 * A saved assessment, opened for reading
 * ------------------------------------------------------------------ */
console.log("\nA saved assessment opens read-only");
const readOnly = entry.render({
  assessment,
  payload: { ...entry.payloadFor(assessment), id: "6a8742e26ae42abe4d1842c6" },
  detail: [],
  readOnly: true,
});
check("it says so", readOnly.includes("read only"));
check("the inputs are not editable", /readonly=/i.test(readOnly));
check("there is no save button", !readOnly.includes("Save assessment"));

await rm(dirname(outfile), { recursive: true, force: true });
console.log(failures ? `\n${failures} check(s) failed` : "\nall checks passed");
process.exit(failures ? 1 : 0);
