/* Clinician interface for the assessment pipeline.
 *
 * No framework and no build step: one fetch to POST /assessments/parse, then
 * render its envelope. The pipeline is the interesting part of this project;
 * the front end should not need a toolchain to read.
 *
 * All text from the server is written with textContent, never innerHTML. The
 * transcript and every extracted value originate in a recording and pass
 * through a language model, so they are untrusted strings.
 */
'use strict';

const $ = (id) => document.getElementById(id);

/* Weights mirror SECTION_WEIGHTS in app/extraction/confidence.py. They are
 * duplicated here only to *explain* the score the server already computed —
 * the number shown always comes from the response, never from this table. */
const WEIGHTS = [
  ['clinicalDetails', 0.30, 'Clinical details'],
  ['objectiveAssessment', 0.25, 'Objective measurements'],
  ['subjectiveAssessments', 0.15, 'Subjective findings'],
  ['recommendation', 0.15, 'Recommendation'],
  ['subjectiveGoals', 0.05, 'Subjective goals'],
  ['objectiveGoals', 0.05, 'Objective goals'],
  ['patientAdvice', 0.05, 'Patient advice'],
];

const REJECTION_PENALTY = 0.10;

/* Measured on the reference recording; used only to animate the stage list
 * while one long request is in flight. Real timings replace these on arrival. */
const STAGES = [
  ['Transcribing audio', 'Whisper, running on the CPU', 29],
  ['Clinical details', '', 14],
  ['Subjective assessment', '', 19],
  ['Objective measurements', '', 41],
  ['Treatment goals', '', 26],
  ['Plan and advice', '', 6],
  ['Verifying every value against the transcript', 'No model involved in this step', 1],
];

let chosenFile = null;
let result = null;
let signed = false;
let ticker = null;

/* ------------------------------------------------------------------ utils */
function el(tag, cls, text) {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text !== undefined) node.textContent = text;
  return node;
}

function fmtBytes(n) {
  if (n < 1024) return n + ' B';
  if (n < 1024 * 1024) return (n / 1024).toFixed(0) + ' KB';
  return (n / (1024 * 1024)).toFixed(1) + ' MB';
}

function fmtDuration(seconds) {
  if (!seconds) return '';
  const m = Math.floor(seconds / 60);
  const s = Math.round(seconds % 60);
  return m ? `${m} min ${s} sec` : `${s} sec`;
}

function drawWave(target, count) {
  target.textContent = '';
  // Deterministic, so the bars do not reshuffle on every re-render.
  for (let i = 0; i < count; i++) {
    const h = 4 + Math.abs(Math.sin(i * 1.7) * Math.cos(i * 0.6)) * 34;
    const bar = el('i');
    bar.style.height = h.toFixed(0) + 'px';
    if (h > 22) bar.className = 'tall';
    target.appendChild(bar);
  }
}

/* ----------------------------------------------------------------- intake */
function chooseFile(file) {
  if (!file) return;
  chosenFile = file;

  $('dropEmpty').hidden = true;
  $('dropFile').hidden = false;
  $('drop').classList.add('has-file');
  $('fileName').textContent = file.name;
  $('fileMeta').textContent = fmtBytes(file.size) + ' · WAV';
  drawWave($('wave'), 44);

  $('btnStart').disabled = false;
  $('startHint').textContent =
    'About two minutes on local models. You can leave this page open.';
  $('intakeError').hidden = true;
}

function renderStages(activeIndex, realTimings) {
  const host = $('stages');
  host.textContent = '';

  STAGES.forEach(([label, note, _est], i) => {
    const done = activeIndex > i;
    const running = activeIndex === i;

    const row = el('div', 'stage ' + (done ? 'is-done' : running ? 'is-running' : 'is-pending'));
    row.appendChild(el('div', 'stage__mark'));

    const mid = el('div', 'stage__label');
    mid.appendChild(document.createTextNode(label));
    if (note && (running || done)) mid.appendChild(el('div', 'stage__note', note));
    row.appendChild(mid);

    // `!= null` covers both null and undefined. `!== undefined` alone let a
    // null timing through and rendered the string "nulls" while processing.
    const key = ['transcribe', 'clinicalDetails', 'subjective', 'objective', 'goals', 'plan', 'grounding'][i];
    const secs = realTimings ? realTimings[key] : null;
    row.appendChild(el('div', 'stage__time', done && secs != null ? secs + 's' : ''));

    host.appendChild(row);
  });
}

function startTicker() {
  const began = Date.now();
  const total = STAGES.reduce((sum, s) => sum + s[2], 0);
  renderStages(0, null);

  ticker = setInterval(() => {
    const secs = (Date.now() - began) / 1000;
    $('elapsed').textContent = fmtDuration(secs);

    let cursor = 0;
    let acc = 0;
    for (let i = 0; i < STAGES.length; i++) {
      acc += STAGES[i][2];
      if (secs < acc) { cursor = i; break; }
      cursor = Math.min(i + 1, STAGES.length - 1);
    }
    renderStages(cursor, null);
    if (secs > total * 2) clearInterval(ticker);   // stop guessing if it overruns
  }, 500);
}

function showError(title, body, fields) {
  const host = $('intakeError');
  host.textContent = '';
  host.hidden = false;

  const box = el('div', 'alert' + (fields ? ' alert--warn' : ''));
  box.appendChild(el('div', 'alert__t', title));
  box.appendChild(el('div', 'alert__b', body));

  if (fields && fields.length) {
    const list = el('div', 'alert__list');
    fields.slice(0, 12).forEach((f) => list.appendChild(el('div', null, f.path + ' — ' + f.reason)));
    box.appendChild(list);
  }
  host.appendChild(box);
}

async function run() {
  if (!chosenFile) return;

  $('btnStart').disabled = true;
  $('btnStart').textContent = 'Working…';
  $('drop').disabled = true;
  $('pipeline').hidden = false;
  $('intakeError').hidden = true;
  startTicker();

  const body = new FormData();
  body.append('file', chosenFile);

  let response;
  try {
    response = await fetch('/assessments/parse', { method: 'POST', body });
  } catch (err) {
    clearInterval(ticker);
    $('pipeline').hidden = true;
    $('btnStart').disabled = false;
    $('btnStart').textContent = 'Begin processing';
    $('drop').disabled = false;
    showError('Could not reach the service',
      'The request failed before the server answered. Check that the API is running, then try again.');
    return;
  }

  clearInterval(ticker);
  const payload = await response.json().catch(() => ({}));

  if (response.status === 422 && payload.detail && typeof payload.detail === 'object') {
    const d = payload.detail;
    $('pipeline').hidden = true;
    $('btnStart').disabled = false;
    $('btnStart').textContent = 'Begin processing';
    $('drop').disabled = false;
    showError(
      'Not enough was captured to build an assessment',
      `Confidence ${Number(d.confidence).toFixed(2)}, below the ${Number(d.threshold).toFixed(2)} ` +
      'threshold. Nothing has been saved. Re-record if you can, or complete the form by hand.',
      d.fields || []
    );
    return;
  }

  if (!response.ok) {
    $('pipeline').hidden = true;
    $('btnStart').disabled = false;
    $('btnStart').textContent = 'Begin processing';
    $('drop').disabled = false;
    const detail = typeof payload.detail === 'string' ? payload.detail : 'The request could not be completed.';
    showError(response.status === 413 ? 'That recording is too large'
      : response.status === 400 ? 'That file could not be read'
      : 'Something went wrong', detail);
    return;
  }

  result = payload;
  renderStages(STAGES.length, result.timings);
  showReview();
}

/* ----------------------------------------------------------------- review */
function fieldRow(label, value, path) {
  const row = el('div', 'field');
  if (path) row.dataset.path = path;
  row.appendChild(el('div', 'field__label', label));

  const v = el('div', 'field__value');
  if (value && String(value).trim()) {
    v.textContent = value;
  } else {
    v.appendChild(el('span', 'blank', 'Not stated in this recording'));
    row.classList.add('is-flagged');
  }
  row.appendChild(v);
  return row;
}

function renderPresentation(a) {
  const host = $('secPresentation');
  host.textContent = '';
  const d = a.clinicalDetails;
  host.appendChild(fieldRow('Chief complaint', d.chiefComplaint, 'clinicalDetails.chiefComplaint'));
  host.appendChild(fieldRow('History', d.clinicalHistory, 'clinicalDetails.clinicalHistory'));
  host.appendChild(fieldRow('Duration', d.duration, 'clinicalDetails.duration'));

  const filled = [d.chiefComplaint, d.clinicalHistory, d.duration].filter((x) => x && x.trim()).length;
  $('presNote').textContent = filled === 3 ? 'Complete' : `${filled} of 3 recorded`;
}

function renderObjective(a) {
  const host = $('secObjective');
  host.textContent = '';
  const tests = a.objectiveAssessment.tests || [];

  if (!tests.length) {
    host.appendChild(fieldRow('Measurements', '', 'objectiveAssessment.tests'));
    return;
  }

  const table = el('table', 'mtable');
  const head = el('tr');
  ['Test', 'Left', 'Right', 'Delta'].forEach((h) => head.appendChild(el('th', null, h)));
  table.appendChild(head);

  tests.forEach((t, i) => {
    const row = el('tr');
    row.dataset.path = `objectiveAssessment.tests[${i}]`;

    row.appendChild(el('td', null, t.testName || '—'));

    const left = t.left || t.value || '';
    const right = t.right || '';
    const l = el('td', 'num', left || '—');
    if (!left) l.classList.add('none');
    const r = el('td', 'num', right || '—');
    if (!right) r.classList.add('none');
    row.appendChild(l);
    row.appendChild(r);

    const lf = parseFloat(left);
    const rf = parseFloat(right);
    const cell = el('td', 'delta');
    if (!isNaN(lf) && !isNaN(rf)) {
      const gap = Math.round((lf - rf) * 10) / 10;
      cell.textContent = gap === 0 ? '—' : (gap > 0 ? '+' + gap : String(gap));
      if (Math.abs(gap) >= 5) cell.classList.add('notable');
    } else {
      cell.textContent = '—';
      cell.classList.add('none');
    }
    row.appendChild(cell);
    table.appendChild(row);
  });

  host.appendChild(table);
}

function renderSubjective(a) {
  const host = $('secSubjective');
  host.textContent = '';
  const items = a.subjectiveAssessments || [];

  if (!items.length) {
    host.appendChild(fieldRow('Findings', '', 'subjectiveAssessments'));
    return;
  }
  items.forEach((s, i) => {
    const row = fieldRow(s.testName || `Finding ${i + 1}`, s.conclusion, `subjectiveAssessments[${i}]`);
    row.querySelector('.field__label').style.textTransform = 'none';
    row.querySelector('.field__label').style.letterSpacing = '0';
    row.querySelector('.field__label').style.fontSize = '13.5px';
    host.appendChild(row);
  });
}

function goalRow(name, date, path) {
  const row = el('div', 'goal');
  row.dataset.path = path;
  row.appendChild(el('div', 'goal__name', name || '—'));
  if (date && date.trim()) {
    row.appendChild(el('div', 'num', date));
  } else {
    row.appendChild(el('div', 'blank blank--sm', 'Not stated'));
  }
  return row;
}

function renderGoals(a) {
  const host = $('secGoals');
  host.textContent = '';

  (a.objectiveGoals || []).forEach((g, i) =>
    host.appendChild(goalRow(g.goalName, g.targetDate, `objectiveGoals[${i}].targetDate`)));
  (a.subjectiveGoals || []).forEach((g, i) =>
    host.appendChild(goalRow(g.goalDetails, g.targetDate, `subjectiveGoals[${i}].targetDate`)));

  if (!(a.objectiveGoals || []).length && !(a.subjectiveGoals || []).length) {
    const row = el('div', 'goal');
    row.dataset.path = 'subjectiveGoals';
    row.appendChild(el('div', 'field__label', 'Goals'));
    row.appendChild(el('div', 'blank blank--sm', 'None recorded'));
    host.appendChild(row);
  }

  const missing = (a.objectiveGoals || []).concat(a.subjectiveGoals || [])
    .filter((g) => !(g.targetDate || '').trim()).length;
  $('goalsNote').textContent = missing ? `${missing} target dates not stated` : '';
}

function renderPlan(a) {
  const host = $('secPlan');
  host.textContent = '';

  const recs = a.recommendation || [];
  const wrap = el('div', 'field');
  wrap.dataset.path = 'recommendation';
  wrap.appendChild(el('div', 'field__label', 'Recommendation'));
  const body = el('div', 'field__value');

  if (recs.length) {
    recs.forEach((r) => {
      body.appendChild(el('div', 'plan__rec', r.sessionType || '—'));
      if (r.sessionFrequency) body.appendChild(el('div', 'plan__freq', r.sessionFrequency));
    });
  } else {
    body.appendChild(el('span', 'blank', 'Not stated in this recording'));
    wrap.classList.add('is-flagged');
  }
  wrap.appendChild(body);
  host.appendChild(wrap);

  host.appendChild(fieldRow('Patient advice', a.patientAdvice.adviceDetails, 'patientAdvice.adviceDetails'));
}

function renderConfidence() {
  const c = result.confidence;
  const pct = Math.max(0, Math.min(1, c.overall)) * 100;

  $('confScore').textContent = c.overall.toFixed(2);
  $('confFill').style.width = pct + '%';
  $('confTick').style.left = (c.threshold * 100) + '%';
  $('conf').classList.toggle('is-below', !c.meetsThreshold);

  $('confVerdict').textContent = c.meetsThreshold
    ? `Above the ${c.threshold.toFixed(2)} threshold required to sign off.`
    : `Below the ${c.threshold.toFixed(2)} threshold.`;

  // Breakdown: how the number was arrived at.
  const grid = $('breakdownGrid');
  grid.textContent = '';
  ['Section', 'Filled', 'Weight', 'Contributes'].forEach((h) =>
    grid.appendChild(el('div', 'h' + (h === 'Section' ? '' : ' num'), h)));

  let weighted = 0;
  WEIGHTS.forEach(([key, weight, label]) => {
    const score = c.sectionScores[key] !== undefined ? c.sectionScores[key] : 0;
    weighted += score * weight;
    grid.appendChild(el('div', null, label));
    grid.appendChild(el('div', 'num', (score * 100).toFixed(0) + '%'));
    grid.appendChild(el('div', 'num', weight.toFixed(2)));
    grid.appendChild(el('div', 'num', (score * weight).toFixed(3)));
  });

  const penalty = Math.min(c.rejectedCount * REJECTION_PENALTY, 0.5);
  const total = $('breakdownTotal');
  total.textContent = '';
  total.appendChild(el('span', null, `Weighted total ${weighted.toFixed(3)}`));
  total.appendChild(el('span', null,
    `Penalty ${penalty ? '−' + penalty.toFixed(2) : '0.00'} (${c.rejectedCount} discarded × 0.10)`));
  total.appendChild(el('span', null, `Confidence ${c.overall.toFixed(2)}`));
}

function renderStats() {
  const a = result.assessment;
  const c = result.confidence;
  const tests = (a.objectiveAssessment.tests || []).length;
  const dates = (a.objectiveGoals || []).concat(a.subjectiveGoals || [])
    .filter((g) => (g.targetDate || '').trim()).length;

  const host = $('stats');
  host.textContent = '';
  [
    [String(tests), 'measurements captured', ''],
    [String(c.rejectedCount), 'values discarded as ungrounded', c.rejectedCount ? 'warn' : 'good'],
    [String(dates), 'dates recorded', dates ? '' : 'good'],
    [String(result.flaggedFields.length), 'fields flagged for review', 'warn'],
  ].forEach(([n, label, tone]) => {
    const box = el('div');
    box.appendChild(el('div', 'n ' + tone, n));
    box.appendChild(el('div', 'l', label));
    host.appendChild(box);
  });
}

function renderFlags() {
  const host = $('flagList');
  host.textContent = '';

  const rejected = result.flaggedFields.filter((f) => f.reason === 'rejected');
  $('tallyDiscarded').textContent = String(result.confidence.rejectedCount);
  $('tallyFlagged').textContent = String(result.flaggedFields.length);

  $('discardedNote').textContent = result.confidence.rejectedCount
    ? 'Values the model produced that could not be traced to the recording. Cleared, and listed below for audit.'
    : 'Every value above was traced back to something actually said in the recording. Nothing was inferred.';

  result.flaggedFields.forEach((f) => {
    const btn = el('button', 'flag' + (f.reason === 'rejected' ? ' flag--rejected' : ''));
    btn.type = 'button';
    btn.appendChild(el('div', 'flag__path', f.path));
    btn.appendChild(el('div', 'flag__why',
      f.reason === 'rejected' ? (f.detail || 'Discarded — not in the recording') : (f.detail || 'Not stated in the recording')));
    btn.addEventListener('click', () => revealField(f.path, btn));
    host.appendChild(btn);
  });

  if (rejected.length) $('tallyDiscarded').parentElement.parentElement.classList.add('tally--warn');
}

function revealField(path, btn) {
  document.querySelectorAll('.flag.is-active').forEach((n) => n.classList.remove('is-active'));
  btn.classList.add('is-active');

  // Match the field itself, or the nearest ancestor path (a goal's targetDate
  // flag should reveal that goal's row).
  let target = document.querySelector(`[data-path="${CSS.escape(path)}"]`);
  if (!target) {
    const base = path.replace(/\.[A-Za-z]+$/, '');
    target = document.querySelector(`[data-path="${CSS.escape(base)}"]`);
  }
  if (!target) return;

  target.scrollIntoView({ behavior: 'smooth', block: 'center' });
  target.classList.remove('is-target');
  void target.offsetWidth;                 // restart the animation
  target.classList.add('is-target');
}

function renderPrintExtras() {
  const a = result.assessment;
  const t = result.transcript;
  const c = result.confidence;

  $('pTitle').textContent = 'First assessment';
  $('pMeta').textContent =
    `${chosenFile ? chosenFile.name : 'recording'} · ${fmtDuration(t.durationSeconds)} · ` +
    `${new Date().toLocaleDateString('en-GB', { day: 'numeric', month: 'long', year: 'numeric' })} · ` +
    `confidence ${c.overall.toFixed(2)} (threshold ${c.threshold.toFixed(2)})`;

  $('pAppendixLede').textContent =
    'These fields were left blank because the recording did not cover them, or ' +
    'because a value could not be traced back to it. They were not inferred. ' +
    'Complete them by hand before filing this record.';

  const host = $('pAppendix');
  host.textContent = '';
  result.flaggedFields.forEach((f) => {
    const row = el('div', 'row');
    row.appendChild(el('div', 'p', f.path));
    row.appendChild(el('div', 'w',
      f.reason === 'rejected'
        ? 'Discarded — ' + (f.detail || 'could not be traced to the recording')
        : 'Not stated in the recording'));
    host.appendChild(row);
  });

  $('pFoot').textContent =
    `Generated from ${chosenFile ? chosenFile.name : 'a recording'} by the Structured Clinical ` +
    `Assessment pipeline (${t.backend} ${t.model}). Every value in this report was verified ` +
    `against the transcript; ${c.rejectedCount} were discarded as untraceable. This document ` +
    `requires clinician review and signature before it forms part of a medical record.`;
}

function showReview() {
  const a = result.assessment;
  const t = result.transcript;

  $('viewIntake').hidden = true;
  $('viewReview').hidden = false;
  $('ctx').textContent = 'Assessment review';
  ['btnTranscript', 'btnPdf', 'btnSign', 'btnNew'].forEach((id) => { $(id).hidden = false; });

  const complaint = (a.clinicalDetails.chiefComplaint || '').trim();
  $('rvTitle').textContent = complaint
    ? complaint.charAt(0).toUpperCase() + complaint.slice(1)
    : 'Assessment from recording';

  const meta = $('rvMeta');
  meta.textContent = '';
  const bits = [
    [chosenFile ? chosenFile.name : 'recording', 'file'],
    [fmtDuration(t.durationSeconds), ''],
    [new Date().toLocaleDateString('en-GB', { day: 'numeric', month: 'long', year: 'numeric' }), ''],
    [`${t.backend} ${t.model}`, ''],
  ];
  bits.forEach(([text, cls], i) => {
    if (i) meta.appendChild(el('span', 'dot'));
    meta.appendChild(el('span', cls, text));
  });

  renderConfidence();
  renderStats();
  renderPresentation(a);
  renderObjective(a);
  renderSubjective(a);
  renderGoals(a);
  renderPlan(a);
  renderFlags();
  renderPrintExtras();

  $('transcriptText').textContent = t.text;
  $('transcriptNote').textContent =
    `${t.text.split(/\s+/).length} words · ${t.segments} segments · transcribed by ${t.backend} ${t.model}.`;

  window.scrollTo({ top: 0 });
}

/* ------------------------------------------------------------------ wiring */
$('drop').addEventListener('click', () => $('picker').click());
$('picker').addEventListener('change', (e) => chooseFile(e.target.files[0]));

['dragenter', 'dragover'].forEach((type) =>
  $('drop').addEventListener(type, (e) => { e.preventDefault(); $('drop').classList.add('is-over'); }));
['dragleave', 'drop'].forEach((type) =>
  $('drop').addEventListener(type, (e) => { e.preventDefault(); $('drop').classList.remove('is-over'); }));
$('drop').addEventListener('drop', (e) => chooseFile(e.dataTransfer.files[0]));

$('btnStart').addEventListener('click', run);

$('btnTranscript').addEventListener('click', () => {
  const box = $('transcript');
  box.hidden = !box.hidden;
  $('btnTranscript').classList.toggle('btn--on', !box.hidden);
});

$('btnHow').addEventListener('click', () => {
  const box = $('breakdown');
  box.hidden = !box.hidden;
  $('btnHow').textContent = box.hidden ? 'How is this calculated?' : 'Hide the calculation';
});

$('btnPdf').addEventListener('click', () => window.print());

$('btnSign').addEventListener('click', () => {
  if (signed) return;                       // one-way: a record is not un-signed
  signed = true;
  $('btnSign').textContent = 'Signed';
  $('btnSign').classList.remove('btn--primary');
  $('btnSign').classList.add('btn--done');
  $('signedBox').hidden = false;
  $('signedDetail').textContent =
    `${new Date().toLocaleString('en-GB')} · ${result.flaggedFields.length} flagged fields reviewed and acknowledged.`;
});

$('btnNew').addEventListener('click', () => window.location.reload());
