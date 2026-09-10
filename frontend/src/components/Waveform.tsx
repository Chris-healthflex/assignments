import React, { useEffect, useRef, useState } from 'react';

interface WaveformProps {
  file: File | null;
}

export const Waveform: React.FC<WaveformProps> = ({ file }) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [duration, setDuration] = useState<number | null>(null);
  const [isDecoding, setIsDecoding] = useState<boolean>(false);

  useEffect(() => {
    if (!file) {
      setDuration(null);
      return;
    }

    let isCancelled = false;
    setIsDecoding(true);

    const reader = new FileReader();
    reader.onload = async (e) => {
      try {
        const arrayBuffer = e.target?.result as ArrayBuffer;
        if (!arrayBuffer) return;

        const audioCtx = new (window.AudioContext || (window as any).webkitAudioContext)();
        const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);

        if (isCancelled) return;

        const dur = audioBuffer.duration;
        setDuration(dur);
        setIsDecoding(false);

        // Render waveform onto canvas
        const canvas = canvasRef.current;
        if (!canvas) return;

        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        // High DPI handling
        const dpr = window.devicePixelRatio || 1;
        const rect = canvas.getBoundingClientRect();
        canvas.width = rect.width * dpr;
        canvas.height = rect.height * dpr;
        ctx.scale(dpr, dpr);

        const width = rect.width;
        const height = rect.height;

        // Clear background
        ctx.fillStyle = '#FFFFFF';
        ctx.fillRect(0, 0, width, height);

        // Center line
        const midY = height / 2;
        ctx.strokeStyle = '#D9DAD3';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(0, midY);
        ctx.lineTo(width, midY);
        ctx.stroke();

        // Extract peaks
        const rawData = audioBuffer.getChannelData(0);
        const step = Math.ceil(rawData.length / width);
        const amp = height / 2 - 12;

        ctx.strokeStyle = '#15181B';
        ctx.lineWidth = 1.5;
        ctx.beginPath();

        for (let i = 0; i < width; i++) {
          let min = 1.0;
          let max = -1.0;
          const startIdx = i * step;
          const endIdx = Math.min(startIdx + step, rawData.length);

          for (let j = startIdx; j < endIdx; j++) {
            const datum = rawData[j];
            if (datum < min) min = datum;
            if (datum > max) max = datum;
          }

          if (max < min) {
            min = 0;
            max = 0;
          }

          ctx.moveTo(i, midY + min * amp);
          ctx.lineTo(i, midY + max * amp);
        }
        ctx.stroke();

        // Draw 10-second interval tick marks
        if (dur > 0) {
          ctx.fillStyle = '#6B7079';
          ctx.font = '10px "IBM Plex Mono", monospace';
          ctx.textAlign = 'center';

          const interval = 10; // 10 seconds
          const numTicks = Math.floor(dur / interval);

          for (let t = 0; t <= numTicks; t++) {
            const sec = t * interval;
            const x = (sec / dur) * width;

            // Tick mark
            ctx.strokeStyle = '#15181B';
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(x, height - 8);
            ctx.lineTo(x, height);
            ctx.stroke();

            // Label
            const mins = Math.floor(sec / 60);
            const secs = sec % 60;
            const label = `${mins}:${secs < 10 ? '0' : ''}${secs}`;
            if (x > 20 && x < width - 20) {
              ctx.fillText(label, x, height - 12);
            }
          }
        }
      } catch (err) {
        console.error('Failed to decode audio for waveform:', err);
        setIsDecoding(false);
      }
    };

    reader.readAsArrayBuffer(file);

    return () => {
      isCancelled = true;
    };
  }, [file]);

  if (!file) return null;

  return (
    <div style={{ marginTop: '16px', marginBottom: '24px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '6px' }}>
        <span style={{ fontSize: '12px', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em' }}>
          WAV Session Audio Waveform
        </span>
        <span className="mono" style={{ fontSize: '12px', color: 'var(--muted)' }}>
          {isDecoding ? 'Decoding samples…' : duration ? `${duration.toFixed(1)}s (16 kHz mono normalized)` : ''}
        </span>
      </div>
      <canvas
        ref={canvasRef}
        className="waveform-canvas"
        style={{ width: '100%', height: '80px' }}
      />
    </div>
  );
};
