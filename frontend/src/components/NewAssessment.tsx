import React, { useState, useRef } from 'react';
import { UploadCloud, CheckCircle2, AlertCircle, Loader2, ArrowRight } from 'lucide-react';
import { Waveform } from './Waveform';
import { parseAudio } from '../api';
import { FirstAssessment, ParseResponseMeta } from '../types/api';

interface NewAssessmentProps {
  onAssessmentParsed: (assessment: FirstAssessment, meta: ParseResponseMeta) => void;
}

const STAGES = [
  { id: 'guard', label: 'Audio Guard & Resample (16kHz Mono)' },
  { id: 'whisper', label: 'Whisper Transcription & VAD Timestamps' },
  { id: 'extract', label: 'LangGraph Extract Node (Structured Output)' },
  { id: 'ground', label: 'Deterministic Numeric & Date Grounding' },
  { id: 'audit', label: 'Audit Fusion & Schema Contract Normalization' },
];

export const NewAssessment: React.FC<NewAssessmentProps> = ({ onAssessmentParsed }) => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isProcessing, setIsProcessing] = useState<boolean>(false);
  const [currentStageIdx, setCurrentStageIdx] = useState<number>(-1);
  const [error, setError] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState<boolean>(false);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      if (!file.name.toLowerCase().endsWith('.wav')) {
        setError('Only WAV audio files (.wav) are accepted by the clinical pipeline.');
        return;
      }
      setSelectedFile(file);
      setError(null);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      if (!file.name.toLowerCase().endsWith('.wav')) {
        setError('Only WAV audio files (.wav) are accepted by the clinical pipeline.');
        return;
      }
      setSelectedFile(file);
      setError(null);
    }
  };

  const handleStartPipeline = async () => {
    if (!selectedFile) return;

    setIsProcessing(true);
    setError(null);
    setCurrentStageIdx(0);

    // Staged progression animation while awaiting server
    const interval = setInterval(() => {
      setCurrentStageIdx((prev) => (prev < STAGES.length - 2 ? prev + 1 : prev));
    }, 2500);

    try {
      const result = await parseAudio(selectedFile);
      clearInterval(interval);
      setCurrentStageIdx(STAGES.length - 1);
      setTimeout(() => {
        setIsProcessing(false);
        onAssessmentParsed(result.assessment, result.meta);
      }, 600);
    } catch (err: any) {
      clearInterval(interval);
      setIsProcessing(false);
      setError(err.message || 'Failed to process clinical assessment audio.');
      if (err.data && err.data.flags) {
        // Returned 422 with confidence report
        setError(`Assessment flagged for review: overall confidence ${(err.data.overall * 100).toFixed(0)}% is below the clinical threshold (${err.data.threshold * 100}%).`);
      }
    }
  };

  return (
    <div style={{ maxWidth: '800px', margin: '0 auto', paddingTop: '16px' }}>
      <div style={{ marginBottom: '24px' }}>
        <h2 style={{ fontSize: '24px', letterSpacing: '-0.02em', marginBottom: '8px' }}>
          Upload Clinical Session Audio
        </h2>
        <p style={{ color: 'var(--muted)', fontSize: '14px' }}>
          Turn a clinician-patient consultation recording into a validated FirstAssessment record conforming to the Stance Health contract.
        </p>
      </div>

      {/* Drag and drop upload box */}
      <div
        onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        onDrop={handleDrop}
        onClick={() => fileInputRef.current?.click()}
        style={{
          border: `2px dashed ${dragOver ? 'var(--ink)' : 'var(--line)'}`,
          backgroundColor: dragOver ? '#FFFFFF' : '#FAFAF8',
          padding: '40px 24px',
          textAlign: 'center',
          cursor: 'pointer',
          borderRadius: '2px',
          transition: 'all 0.15s ease',
        }}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept=".wav,audio/wav"
          onChange={handleFileChange}
          style={{ display: 'none' }}
        />
        <UploadCloud size={36} color="#6B7079" style={{ margin: '0 auto 12px auto' }} />
        <h3 style={{ fontSize: '16px', marginBottom: '6px' }}>
          {selectedFile ? selectedFile.name : 'Select or drop clinical audio WAV file'}
        </h3>
        <p style={{ fontSize: '13px', color: 'var(--muted)' }}>
          {selectedFile
            ? `${(selectedFile.size / (1024 * 1024)).toFixed(2)} MB · Click or drop to replace`
            : 'Standard PCM WAV mono/stereo up to 50MB'}
        </p>
      </div>

      {/* Audio Waveform Renderer */}
      {selectedFile && <Waveform file={selectedFile} />}

      {/* Pipeline Run Button */}
      {selectedFile && !isProcessing && (
        <div style={{ marginTop: '20px', display: 'flex', justifyContent: 'flex-end' }}>
          <button className="btn btn-primary" onClick={handleStartPipeline} style={{ padding: '12px 24px' }}>
            Process Recording Through Pipeline
            <ArrowRight size={16} />
          </button>
        </div>
      )}

      {/* Live Stage Progress Indicator */}
      {isProcessing && (
        <div style={{ marginTop: '28px', borderTop: '1px solid var(--line)', paddingTop: '20px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '16px' }}>
            <Loader2 className="animate-spin" size={18} color="var(--ink)" />
            <span style={{ fontSize: '14px', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.04em' }}>
              Processing Pipeline In Flight…
            </span>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
            {STAGES.map((stage, idx) => {
              const isDone = idx < currentStageIdx;
              const isCurrent = idx === currentStageIdx;

              return (
                <div
                  key={stage.id}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '12px',
                    padding: '8px 12px',
                    backgroundColor: isCurrent ? '#FFFFFF' : 'transparent',
                    border: isCurrent ? '1px solid var(--ink)' : '1px solid transparent',
                    borderRadius: '2px',
                  }}
                >
                  {isDone ? (
                    <CheckCircle2 size={16} color="var(--verified)" />
                  ) : isCurrent ? (
                    <Loader2 className="animate-spin" size={16} color="var(--ink)" />
                  ) : (
                    <div style={{ width: '16px', height: '16px', borderRadius: '50%', border: '1px solid var(--line)' }} />
                  )}
                  <span
                    style={{
                      fontSize: '13px',
                      color: isCurrent ? 'var(--ink)' : isDone ? 'var(--verified)' : 'var(--muted)',
                      fontWeight: isCurrent ? 600 : 400,
                    }}
                  >
                    {stage.label}
                  </span>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Error / Alert Display */}
      {error && (
        <div
          style={{
            marginTop: '24px',
            backgroundColor: 'var(--flag-bg)',
            border: '1px solid var(--flag)',
            padding: '16px',
            display: 'flex',
            gap: '12px',
            alignItems: 'flex-start',
          }}
        >
          <AlertCircle size={20} color="var(--flag)" style={{ flexShrink: 0, marginTop: '2px' }} />
          <div>
            <h4 style={{ color: 'var(--flag)', fontSize: '14px', marginBottom: '4px' }}>
              Pipeline Notice
            </h4>
            <p style={{ fontSize: '13px', color: 'var(--ink)' }}>{error}</p>
          </div>
        </div>
      )}
    </div>
  );
};
