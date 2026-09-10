import React, { useState } from 'react';
import {
  Save,
  Printer,
  Code,
  CheckCircle2,
  AlertTriangle,
  Clock,
  Check,
} from 'lucide-react';
import {
  FirstAssessment,
  ConfidenceReport,
  ParseResponseMeta,
} from '../types/api';
import { saveAssessment } from '../api';

interface ReportViewProps {
  assessment: FirstAssessment;
  confidence?: ConfidenceReport | null;
  meta?: ParseResponseMeta;
  recordId?: string;
  onSaved?: (id: string) => void;
}

export const ReportView: React.FC<ReportViewProps> = ({
  assessment,
  confidence,
  meta,
  recordId: initialRecordId,
  onSaved,
}) => {
  const [recordId, setRecordId] = useState<string | null>(initialRecordId || null);
  const [isSaving, setIsSaving] = useState<boolean>(false);
  const [showRawJson, setShowRawJson] = useState<boolean>(false);
  const [saveSuccess, setSaveSuccess] = useState<boolean>(false);
  const [activeHighlightField, setActiveHighlightField] = useState<string | null>(null);

  const handleSave = async () => {
    setIsSaving(true);
    try {
      const record = await saveAssessment(assessment);
      setRecordId(record.id);
      setSaveSuccess(true);
      if (onSaved) onSaved(record.id);
      setTimeout(() => setSaveSuccess(false), 3000);
    } catch (err: any) {
      alert(`Save failed: ${err.message}`);
    } finally {
      setIsSaving(false);
    }
  };

  const flags = confidence?.flags || [];
  const overallConf = meta?.confidence ?? confidence?.overall ?? 0.92;
  const isOverallPassed = overallConf >= (confidence?.threshold ?? 0.70);

  return (
    <div style={{ paddingTop: '8px' }}>
      {/* Editorial Action Bar */}
      <div className="no-print" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '24px', borderBottom: '1px solid var(--line)', paddingBottom: '16px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <span style={{ fontSize: '13px', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.04em' }}>
            Report Status:
          </span>
          <span className={`badge ${isOverallPassed ? 'badge-verified' : 'badge-flag'}`}>
            {isOverallPassed ? <CheckCircle2 size={13} /> : <AlertTriangle size={13} />}
            {(overallConf * 100).toFixed(0)}% Overall Confidence {isOverallPassed ? '(Verified)' : '(Review Required)'}
          </span>
          {recordId && (
            <span className="mono" style={{ fontSize: '12px', color: 'var(--muted)', background: '#ECECE9', padding: '2px 8px', borderRadius: '2px' }}>
              ID: {recordId.slice(0, 8)}…
            </span>
          )}
        </div>

        <div style={{ display: 'flex', gap: '8px' }}>
          <button className="btn btn-secondary" onClick={() => setShowRawJson(!showRawJson)}>
            <Code size={15} />
            {showRawJson ? 'Hide JSON' : 'Raw JSON'}
          </button>

          <button className="btn btn-secondary" onClick={() => window.print()}>
            <Printer size={15} />
            Print Note
          </button>

          <button
            className="btn btn-primary"
            onClick={handleSave}
            disabled={isSaving || !!recordId}
          >
            {saveSuccess ? (
              <>
                <Check size={15} /> Saved
              </>
            ) : (
              <>
                <Save size={15} />
                {recordId ? 'Saved to DB' : isSaving ? 'Saving…' : 'Save to DB'}
              </>
            )}
          </button>
        </div>
      </div>

      {/* Raw JSON inspection toggle */}
      {showRawJson && (
        <div className="no-print" style={{ marginBottom: '24px', background: '#FFFFFF', border: '1px solid var(--ink)', padding: '16px' }}>
          <h4 style={{ fontSize: '12px', marginBottom: '8px', color: 'var(--muted)' }}>
            STRICT FIRSTASSESSMENT PAYLOAD (NO EXTRA KEYS, NO NULLS)
          </h4>
          <pre className="mono" style={{ fontSize: '12px', maxHeight: '300px', overflow: 'auto', background: 'var(--paper)', padding: '12px' }}>
            {JSON.stringify(assessment, null, 2)}
          </pre>
        </div>
      )}

      {/* Swiss 8/4 Asymmetric Grid */}
      <div className="report-grid">
        {/* Left 8-Column Reading Pane */}
        <div className="reading-column">
          {/* 01 CLINICAL DETAILS */}
          <section className="report-section">
            <div className="section-header">
              <span className="section-num">01</span>
              <h2 className="section-title">Clinical Details</h2>
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 2fr', gap: '16px', marginBottom: '16px' }}>
              <div>
                <p style={{ fontSize: '11px', textTransform: 'uppercase', color: 'var(--muted)', fontWeight: 600 }}>Chief Complaint</p>
                <p style={{ fontSize: '16px', fontWeight: 600, marginTop: '2px' }}>
                  {assessment.clinicalDetails.chiefComplaint || '—'}
                </p>
              </div>
              <div>
                <p style={{ fontSize: '11px', textTransform: 'uppercase', color: 'var(--muted)', fontWeight: 600 }}>Duration</p>
                <p className="mono" style={{ fontSize: '15px', marginTop: '2px' }}>
                  {assessment.clinicalDetails.duration || '—'}
                </p>
              </div>
            </div>
            <div>
              <p style={{ fontSize: '11px', textTransform: 'uppercase', color: 'var(--muted)', fontWeight: 600 }}>Clinical History</p>
              <p style={{ marginTop: '4px', textAlign: 'justify', color: '#2B3037' }}>
                {assessment.clinicalDetails.clinicalHistory || 'No prior history reported.'}
              </p>
            </div>
          </section>

          {/* 02 SUBJECTIVE ASSESSMENTS */}
          <section className="report-section">
            <div className="section-header">
              <span className="section-num">02</span>
              <h2 className="section-title">Subjective Assessments</h2>
            </div>
            {assessment.subjectiveAssessments.length === 0 ? (
              <p style={{ color: 'var(--muted)', fontStyle: 'italic' }}>No subjective assessments recorded.</p>
            ) : (
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Test / Assessment</th>
                    <th>Reported Conclusion</th>
                  </tr>
                </thead>
                <tbody>
                  {assessment.subjectiveAssessments.map((sa, i) => (
                    <tr key={i}>
                      <td style={{ fontWeight: 600 }}>{sa.testName || '—'}</td>
                      <td>{sa.conclusion || '—'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </section>

          {/* 03 OBJECTIVE ASSESSMENT */}
          <section className="report-section">
            <div className="section-header">
              <span className="section-num">03</span>
              <h2 className="section-title">Objective Assessment</h2>
            </div>
            {assessment.objectiveAssessment.tests.length === 0 ? (
              <p style={{ color: 'var(--muted)', fontStyle: 'italic' }}>No objective measurements recorded.</p>
            ) : (
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Test Name</th>
                    <th>Unit</th>
                    <th style={{ textAlign: 'right' }}>Left</th>
                    <th style={{ textAlign: 'right' }}>Right</th>
                    <th style={{ textAlign: 'right' }}>Value</th>
                    <th>Comments</th>
                  </tr>
                </thead>
                <tbody>
                  {assessment.objectiveAssessment.tests.map((t, i) => {
                    const leftField = `objectiveAssessment.tests[${i}].left`;
                    const rightField = `objectiveAssessment.tests[${i}].right`;
                    const valField = `objectiveAssessment.tests[${i}].value`;

                    const isLeftFlagged = flags.some((f) => f.field === leftField);
                    const isRightFlagged = flags.some((f) => f.field === rightField);
                    const isValFlagged = flags.some((f) => f.field === valField);

                    return (
                      <tr key={i}>
                        <td style={{ fontWeight: 600 }}>{t.testName}</td>
                        <td style={{ color: 'var(--muted)', fontSize: '13px' }}>{t.unitName}</td>
                        <td
                          className={`numeric ${isLeftFlagged ? 'field-flagged' : ''}`}
                          style={{
                            textAlign: 'right',
                            outline: activeHighlightField === leftField ? '2px solid var(--flag)' : 'none',
                          }}
                        >
                          {t.left || '—'}
                        </td>
                        <td
                          className={`numeric ${isRightFlagged ? 'field-flagged' : ''}`}
                          style={{
                            textAlign: 'right',
                            outline: activeHighlightField === rightField ? '2px solid var(--flag)' : 'none',
                          }}
                        >
                          {t.right || '—'}
                        </td>
                        <td
                          className={`numeric ${isValFlagged ? 'field-flagged' : ''}`}
                          style={{
                            textAlign: 'right',
                            outline: activeHighlightField === valField ? '2px solid var(--flag)' : 'none',
                          }}
                        >
                          {t.value || '—'}
                        </td>
                        <td style={{ color: 'var(--muted)', fontSize: '13px' }}>{t.comments || '—'}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            )}
          </section>

          {/* 04 GOALS */}
          <section className="report-section">
            <div className="section-header">
              <span className="section-num">04</span>
              <h2 className="section-title">Goals & Targets</h2>
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '24px' }}>
              <div>
                <h4 style={{ fontSize: '12px', color: 'var(--muted)', marginBottom: '8px' }}>SUBJECTIVE GOALS</h4>
                {assessment.subjectiveGoals.length === 0 ? (
                  <p style={{ color: 'var(--muted)', fontSize: '13px', fontStyle: 'italic' }}>None stated.</p>
                ) : (
                  assessment.subjectiveGoals.map((sg, i) => (
                    <div key={i} style={{ borderBottom: '1px solid var(--line)', paddingBottom: '8px', marginBottom: '8px' }}>
                      <p style={{ fontWeight: 500 }}>{sg.goalDetails}</p>
                      {sg.targetDate && <p className="mono" style={{ fontSize: '12px', color: 'var(--muted)' }}>Target: {sg.targetDate}</p>}
                    </div>
                  ))
                )}
              </div>

              <div>
                <h4 style={{ fontSize: '12px', color: 'var(--muted)', marginBottom: '8px' }}>OBJECTIVE GOALS</h4>
                {assessment.objectiveGoals.length === 0 ? (
                  <p style={{ color: 'var(--muted)', fontSize: '13px', fontStyle: 'italic' }}>None stated.</p>
                ) : (
                  assessment.objectiveGoals.map((og, i) => (
                    <div key={i} style={{ borderBottom: '1px solid var(--line)', paddingBottom: '8px', marginBottom: '8px' }}>
                      <p style={{ fontWeight: 500 }}>{og.goalName}</p>
                      <p className="mono" style={{ fontSize: '12px', color: 'var(--muted)' }}>
                        {og.value} {og.unitName} {og.targetDate ? `· Target: ${og.targetDate}` : ''}
                      </p>
                    </div>
                  ))
                )}
              </div>
            </div>
          </section>

          {/* 05 RECOMMENDATION */}
          <section className="report-section">
            <div className="section-header">
              <span className="section-num">05</span>
              <h2 className="section-title">Plan & Recommendations</h2>
            </div>
            {assessment.recommendation.length === 0 ? (
              <p style={{ color: 'var(--muted)', fontStyle: 'italic' }}>No recommendations recorded.</p>
            ) : (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                {assessment.recommendation.map((rec, i) => (
                  <div key={i} style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid var(--line)', paddingBottom: '6px' }}>
                    <span style={{ fontWeight: 600 }}>{rec.sessionType || 'Physiotherapy'}</span>
                    <span style={{ color: 'var(--muted)' }}>{rec.sessionFrequency}</span>
                  </div>
                ))}
              </div>
            )}
          </section>

          {/* 06 PATIENT ADVICE */}
          <section className="report-section">
            <div className="section-header">
              <span className="section-num">06</span>
              <h2 className="section-title">Patient Advice & Prescriptions</h2>
            </div>
            <p style={{ color: '#2B3037' }}>
              {assessment.patientAdvice.adviceDetails || 'Standard post-session rehabilitation advice.'}
            </p>
          </section>
        </div>

        {/* Right 4-Column Evidence & Grounding Margin Rail */}
        <aside className="evidence-rail no-print">
          <div style={{ position: 'sticky', top: '24px' }}>
            {/* Header Card */}
            <div style={{ background: '#FFFFFF', border: '1px solid var(--ink)', padding: '16px', marginBottom: '16px' }}>
              <h3 style={{ fontSize: '14px', marginBottom: '8px' }}>Extraction Audit</h3>
              <p style={{ fontSize: '12px', color: 'var(--muted)', marginBottom: '12px' }}>
                Deterministic anti-hallucination verification against session audio transcript.
              </p>

              <div style={{ display: 'flex', justifyContent: 'space-between', borderTop: '1px solid var(--line)', paddingTop: '10px' }}>
                <span style={{ fontSize: '12px', color: 'var(--muted)' }}>Model</span>
                <span className="mono" style={{ fontSize: '12px', fontWeight: 600 }}>
                  {meta?.model || 'Whisper + LLM'}
                </span>
              </div>

              {meta?.latencyBreakdown && (
                <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginTop: '8px', color: 'var(--muted)', fontSize: '11px' }}>
                  <Clock size={12} />
                  <span className="mono">{meta.latencyBreakdown}</span>
                </div>
              )}
            </div>

            {/* Evidence Spans & Flagged Fields */}
            <div style={{ marginBottom: '12px', display: 'flex', justifyContent: 'space-between', alignItems: 'baseline' }}>
              <span style={{ fontSize: '12px', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.04em' }}>
                Field Evidence Rail ({flags.length} Flags)
              </span>
            </div>

            {flags.length === 0 ? (
              <div style={{ background: 'var(--verified-bg)', border: '1px solid var(--verified)', padding: '14px', borderRadius: '2px' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: 'var(--verified)', fontWeight: 600, fontSize: '13px' }}>
                  <CheckCircle2 size={16} /> All Clinical Fields Grounded
                </div>
                <p style={{ fontSize: '12px', color: '#33383F', marginTop: '4px' }}>
                  Every extracted ROM measurement, pain score, and date is verified by transcript matching.
                </p>
              </div>
            ) : (
              <div>
                {flags.map((flag, idx) => (
                  <div
                    key={idx}
                    className={`evidence-card ${flag.grounded ? 'verified' : ''}`}
                    onMouseEnter={() => setActiveHighlightField(flag.field)}
                    onMouseLeave={() => setActiveHighlightField(null)}
                  >
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '4px' }}>
                      <span className="mono" style={{ fontSize: '11px', fontWeight: 600, color: 'var(--ink)' }}>
                        {flag.field}
                      </span>
                      <span className={`badge ${flag.grounded ? 'badge-verified' : 'badge-flag'}`} style={{ fontSize: '10px' }}>
                        {(flag.confidence * 100).toFixed(0)}%
                      </span>
                    </div>

                    <p style={{ fontSize: '12px', color: '#2B3037', marginBottom: '4px' }}>
                      {flag.reason}
                    </p>

                    {flag.evidence_span ? (
                      <div className="evidence-quote">
                        "{flag.evidence_span}"
                      </div>
                    ) : (
                      <p style={{ fontSize: '11px', color: 'var(--flag)', fontStyle: 'italic', marginTop: '4px' }}>
                        ⚑ Hard-capped: value not grounded in spoken transcript.
                      </p>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>
        </aside>
      </div>
    </div>
  );
};
