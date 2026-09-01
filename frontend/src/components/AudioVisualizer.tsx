import React, { useEffect, useRef, useState } from 'react';
import { forensicEngine } from '../utils/audioEngine';
import { Activity, BarChart2, Radio } from 'lucide-react';

interface AudioVisualizerProps {
  isActive: boolean;
  isRecording?: boolean;
  label?: string;
}

export type VisualizerMode = 'waveform' | 'spectrum' | 'phase';

export const AudioVisualizer: React.FC<AudioVisualizerProps> = ({
  isActive,
  isRecording = false,
  label = 'AUDIO STREAM'
}) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [mode, setMode] = useState<VisualizerMode>('waveform');
  const [peakDb, setPeakDb] = useState<string>('-∞ dB');
  const [activeFreq, setActiveFreq] = useState<string>('0 Hz');
  const animationFrameId = useRef<number | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Handle high DPI
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);

    const analyser = forensicEngine.getAnalyser();
    const bufferLength = analyser.frequencyBinCount;
    const timeData = new Uint8Array(bufferLength);
    const freqData = new Uint8Array(bufferLength);

    let phaseOffset = 0;

    const render = () => {
      animationFrameId.current = requestAnimationFrame(render);

      const width = rect.width;
      const height = rect.height;

      // Warm cream sleek background with subtle grid
      ctx.fillStyle = '#FDF8F3';
      ctx.fillRect(0, 0, width, height);

      // Draw subtle horizontal center gridline & dB scale markers
      ctx.strokeStyle = 'rgba(92, 20, 26, 0.06)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(0, height / 2);
      ctx.lineTo(width, height / 2);
      ctx.moveTo(0, height / 4);
      ctx.lineTo(width, height / 4);
      ctx.moveTo(0, (height * 3) / 4);
      ctx.lineTo(width, (height * 3) / 4);
      ctx.stroke();

      if (isActive) {
        analyser.getByteTimeDomainData(timeData);
        analyser.getByteFrequencyData(freqData);

        // Compute peak RMS & peak frequency
        let sumSquares = 0;
        let maxFreqVal = 0;
        let maxFreqIndex = 0;

        for (let i = 0; i < bufferLength; i++) {
          const norm = (timeData[i] - 128) / 128;
          sumSquares += norm * norm;

          if (freqData[i] > maxFreqVal) {
            maxFreqVal = freqData[i];
            maxFreqIndex = i;
          }
        }

        const rms = Math.sqrt(sumSquares / bufferLength);
        const db = rms > 0 ? (20 * Math.log10(rms)).toFixed(1) : '-∞';
        setPeakDb(`${db} dBFS`);

        const nyquist = (forensicEngine.getAudioContext()?.sampleRate || 48000) / 2;
        const dominantFreq = Math.round((maxFreqIndex / bufferLength) * nyquist);
        setActiveFreq(`${dominantFreq} Hz`);
      } else {
        setPeakDb('-∞ dBFS');
        setActiveFreq('0 Hz');
      }

      if (mode === 'waveform') {
        // --- 1. WAVEFORM OSCILLOSCOPE ---
        ctx.lineWidth = 2;
        ctx.strokeStyle = isRecording ? '#5C141A' : 'rgba(92, 20, 26, 0.85)';
        ctx.shadowColor = isRecording ? 'rgba(92, 20, 26, 0.35)' : 'rgba(92, 20, 26, 0.15)';
        ctx.shadowBlur = 6;
        ctx.beginPath();

        const sliceWidth = width / bufferLength;
        let x = 0;

        for (let i = 0; i < bufferLength; i++) {
          let v = 0.5;
          if (isActive) {
            v = timeData[i] / 255.0;
          } else {
            // Idle subtle breathing wave
            v = 0.5 + Math.sin(i * 0.05 + phaseOffset) * 0.02;
          }

          const y = v * height;

          if (i === 0) {
            ctx.moveTo(x, y);
          } else {
            ctx.lineTo(x, y);
          }

          x += sliceWidth;
        }

        ctx.lineTo(width, height / 2);
        ctx.stroke();
        ctx.shadowBlur = 0; // Reset shadow

      } else if (mode === 'spectrum') {
        // --- 2. FFT SPECTRAL BARS ---
        const barCount = 48;
        const barWidth = (width / barCount) - 2;
        const step = Math.floor(bufferLength / barCount);

        for (let i = 0; i < barCount; i++) {
          let barHeight = 4;
          if (isActive) {
            const val = freqData[i * step] || 0;
            barHeight = Math.max(4, (val / 255) * (height - 20));
          } else {
            barHeight = 4 + Math.sin(i * 0.2 + phaseOffset) * 2;
          }

          const x = i * (barWidth + 2);
          const y = height - barHeight;

          // Gradient from deep burgundy to warm rose/terracotta
          const grad = ctx.createLinearGradient(0, height, 0, y);
          grad.addColorStop(0, '#5C141A');
          grad.addColorStop(1, '#B85D64');

          ctx.fillStyle = grad;
          ctx.fillRect(x, y, barWidth, barHeight);
        }

      } else if (mode === 'phase') {
        // --- 3. HARMONIC PHASE LISSAJOUS / ORBIT ---
        const centerX = width / 2;
        const centerY = height / 2;
        const radius = Math.min(width, height) * 0.38;

        ctx.lineWidth = 1.5;
        ctx.strokeStyle = '#5C141A';
        ctx.shadowColor = 'rgba(92, 20, 26, 0.25)';
        ctx.shadowBlur = 6;
        ctx.beginPath();

        const samples = 180;
        for (let i = 0; i < samples; i++) {
          const angle = (i / samples) * Math.PI * 2;
          let amp = 1;

          if (isActive) {
            const dataIdx = Math.floor((i / samples) * (bufferLength / 2));
            const freqVal = (freqData[dataIdx] || 0) / 255.0;
            amp = 0.8 + freqVal * 0.6;
          } else {
            amp = 0.95 + Math.sin(angle * 4 + phaseOffset) * 0.05;
          }

          const r = radius * amp;
          const px = centerX + Math.cos(angle) * r;
          const py = centerY + Math.sin(angle) * r;

          if (i === 0) {
            ctx.moveTo(px, py);
          } else {
            ctx.lineTo(px, py);
          }
        }
        ctx.closePath();
        ctx.stroke();
        ctx.shadowBlur = 0;
      }

      phaseOffset += 0.03;
    };

    render();

    return () => {
      if (animationFrameId.current) {
        cancelAnimationFrame(animationFrameId.current);
      }
    };
  }, [isActive, isRecording, mode]);

  return (
    <div className="visualizer-card" style={{ width: '100%' }}>
      {/* Top DSP readout overlay */}
      <div className="visualizer-overlay-info">
        <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
          <span className="dsp-metric-pill">
            <span className={isActive ? 'status-dot-active' : ''} style={{
              width: 6,
              height: 6,
              borderRadius: '50%',
              backgroundColor: isActive ? 'var(--accent-teal)' : 'var(--text-dim)',
              display: 'inline-block'
            }} />
            <span className="font-mono">{label}</span>
          </span>
          <span className="dsp-metric-pill font-mono">
            PEAK: <span className="metric-val">{peakDb}</span>
          </span>
        </div>

        <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
          <span className="dsp-metric-pill font-mono">
            FREQ: <span className="metric-val">{activeFreq}</span>
          </span>
          <span className="dsp-metric-pill font-mono">
            RATE: <span className="metric-val">48.0 kHz</span>
          </span>
        </div>
      </div>

      {/* Canvas */}
      <canvas
        ref={canvasRef}
        className="visualizer-canvas"
        style={{ width: '100%', height: '180px' }}
      />

      {/* Mode switch buttons */}
      <div className="vis-mode-toggle">
        <button
          type="button"
          onClick={() => setMode('waveform')}
          className={`vis-mode-btn ${mode === 'waveform' ? 'active' : ''}`}
          title="Waveform Oscilloscope"
        >
          <Activity size={12} style={{ display: 'inline', marginRight: 3 }} />
          Wave
        </button>
        <button
          type="button"
          onClick={() => setMode('spectrum')}
          className={`vis-mode-btn ${mode === 'spectrum' ? 'active' : ''}`}
          title="FFT Spectrum"
        >
          <BarChart2 size={12} style={{ display: 'inline', marginRight: 3 }} />
          FFT
        </button>
        <button
          type="button"
          onClick={() => setMode('phase')}
          className={`vis-mode-btn ${mode === 'phase' ? 'active' : ''}`}
          title="Harmonic Phase"
        >
          <Radio size={12} style={{ display: 'inline', marginRight: 3 }} />
          Phase
        </button>
      </div>
    </div>
  );
};
