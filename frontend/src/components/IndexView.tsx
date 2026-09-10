import React, { useEffect, useState } from 'react';
import { RefreshCw, Calendar, ArrowUpRight, CheckCircle2, AlertTriangle } from 'lucide-react';
import { AssessmentRecord } from '../types/api';
import { listAssessments } from '../api';

interface IndexViewProps {
  onSelectRecord: (record: AssessmentRecord) => void;
}

export const IndexView: React.FC<IndexViewProps> = ({ onSelectRecord }) => {
  const [records, setRecords] = useState<AssessmentRecord[]>([]);
  const [total, setTotal] = useState<number>(0);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [fromDate, setFromDate] = useState<string>('');
  const [toDate, setToDate] = useState<string>('');

  const fetchRecords = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const fromIso = fromDate ? new Date(fromDate).toISOString() : undefined;
      const toIso = toDate ? new Date(toDate).toISOString() : undefined;
      const res = await listAssessments(fromIso, toIso, 20, 0);
      setRecords(res.items);
      setTotal(res.total);
    } catch (err: any) {
      setError(err.message || 'Failed to load assessments from MongoDB.');
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchRecords();
  }, []);

  return (
    <div style={{ paddingTop: '8px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-end', marginBottom: '24px', borderBottom: '1px solid var(--line)', paddingBottom: '16px' }}>
        <div>
          <h2 style={{ fontSize: '20px', letterSpacing: '-0.02em', marginBottom: '4px' }}>
            Assessment Registry
          </h2>
          <p style={{ fontSize: '13px', color: 'var(--muted)' }}>
            {total} persisted clinical assessments in MongoDB collection
          </p>
        </div>

        {/* Date Filter Bar */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px', background: '#FFFFFF', border: '1px solid var(--line)', padding: '4px 8px', borderRadius: '2px' }}>
            <Calendar size={14} color="var(--muted)" />
            <input
              type="date"
              value={fromDate}
              onChange={(e) => setFromDate(e.target.value)}
              style={{ border: 'none', background: 'transparent', fontSize: '12px', fontFamily: 'var(--font-mono)' }}
            />
            <span style={{ color: 'var(--muted)', fontSize: '12px' }}>to</span>
            <input
              type="date"
              value={toDate}
              onChange={(e) => setToDate(e.target.value)}
              style={{ border: 'none', background: 'transparent', fontSize: '12px', fontFamily: 'var(--font-mono)' }}
            />
          </div>

          <button className="btn btn-secondary" onClick={fetchRecords} disabled={isLoading}>
            <RefreshCw size={14} className={isLoading ? 'animate-spin' : ''} />
            Filter
          </button>
        </div>
      </div>

      {error && (
        <div style={{ padding: '12px', background: 'var(--flag-bg)', border: '1px solid var(--flag)', marginBottom: '16px', fontSize: '13px', color: 'var(--flag)' }}>
          {error}
        </div>
      )}

      {isLoading ? (
        <div style={{ padding: '40px', textAlign: 'center', color: 'var(--muted)' }}>
          Loading registry records…
        </div>
      ) : records.length === 0 ? (
        <div style={{ padding: '60px 20px', textAlign: 'center', background: '#FFFFFF', border: '1px solid var(--line)' }}>
          <h3 style={{ fontSize: '15px', marginBottom: '6px' }}>No Assessments Found</h3>
          <p style={{ fontSize: '13px', color: 'var(--muted)' }}>
            Upload and process an audio session to persist your first assessment record.
          </p>
        </div>
      ) : (
        <table className="data-table" style={{ background: '#FFFFFF', border: '1px solid var(--line)' }}>
          <thead>
            <tr>
              <th>Date & Time</th>
              <th>Chief Complaint</th>
              <th>Duration</th>
              <th>Tests</th>
              <th>Audit Status</th>
              <th style={{ textAlign: 'right' }}>Action</th>
            </tr>
          </thead>
          <tbody>
            {records.map((r) => {
              const dateStr = new Date(r.createdAt).toLocaleDateString('en-GB', {
                day: '2-digit',
                month: 'short',
                year: 'numeric',
                hour: '2-digit',
                minute: '2-digit',
              });
              const testCount = r.assessment.objectiveAssessment?.tests?.length || 0;
              const overallConf = r.confidence?.overall ?? 0.95;
              const isPassed = overallConf >= (r.confidence?.threshold ?? 0.70);
              const flagCount = r.confidence?.flags?.length || 0;

              return (
                <tr key={r.id} style={{ cursor: 'pointer' }} onClick={() => onSelectRecord(r)}>
                  <td className="mono" style={{ fontSize: '12px', whiteSpace: 'nowrap' }}>
                    {dateStr}
                  </td>
                  <td style={{ fontWeight: 600 }}>
                    {r.assessment.clinicalDetails?.chiefComplaint || 'Clinical Consultation'}
                  </td>
                  <td className="mono" style={{ fontSize: '13px', color: 'var(--muted)' }}>
                    {r.assessment.clinicalDetails?.duration || '—'}
                  </td>
                  <td className="mono" style={{ fontSize: '13px' }}>
                    {testCount} tests
                  </td>
                  <td>
                    <span className={`badge ${isPassed ? 'badge-verified' : 'badge-flag'}`} style={{ fontSize: '11px' }}>
                      {isPassed ? <CheckCircle2 size={12} /> : <AlertTriangle size={12} />}
                      {(overallConf * 100).toFixed(0)}% {flagCount > 0 ? `(${flagCount} flags)` : 'Verified'}
                    </span>
                  </td>
                  <td style={{ textAlign: 'right' }}>
                    <button
                      className="btn btn-secondary"
                      style={{ padding: '4px 10px', fontSize: '11px' }}
                      onClick={(e) => {
                        e.stopPropagation();
                        onSelectRecord(r);
                      }}
                    >
                      Open <ArrowUpRight size={13} />
                    </button>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      )}
    </div>
  );
};
