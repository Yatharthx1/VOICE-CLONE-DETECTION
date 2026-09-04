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
  const peakRef = useRef<HTMLSpanElement | null>(null);
  const freqRef = useRef<HTMLSpanElement | null>(null);
  const lastMetricsUpdate = useRef<number>(0);
  const [mode, setMode] = useState<VisualizerMode>('waveform');
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
    let lastFrameTime = 0;

    const render = (time: number) => {
      if (isActive) {
        animationFrameId.current = requestAnimationFrame(render);
      }

      // Throttle rendering to max 26 FPS to prevent GPU/DWM overload during screen capture
      if (time - lastFrameTime < 38) {
        return;
      }
      lastFrameTime = time;

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

        const now = performance.now();
        if (now - lastMetricsUpdate.current > 150) {
          lastMetricsUpdate.current = now;

          let sumSquares = 0;
          let maxFreqVal = 0;
          let maxFreqIndex = 0;

          // Sample every 4th bin for peak metric
          for (let i = 0; i < bufferLength; i += 4) {
            const norm = (timeData[i] - 128) / 128;
            sumSquares += norm * norm;

            if (freqData[i] > maxFreqVal) {
              maxFreqVal = freqData[i];
              maxFreqIndex = i;
            }
          }

          if (peakRef.current) {
            const rms = Math.sqrt((sumSquares * 4) / bufferLength);
            const db = rms > 0 ? (20 * Math.log10(rms)).toFixed(1) : '-∞';
            peakRef.current.textContent = `${db} dBFS`;
          }

          if (freqRef.current) {
            const nyquist = (forensicEngine.getAudioContext()?.sampleRate || 48000) / 2;
            const dominantFreq = Math.round((maxFreqIndex / bufferLength) * nyquist);
            freqRef.current.textContent = `${dominantFreq} Hz`;
          }
        }
      } else {
        if (peakRef.current) peakRef.current.textContent = '-∞ dBFS';
        if (freqRef.current) freqRef.current.textContent = '0 Hz';
      }

      if (mode === 'waveform') {
        // --- 1. WAVEFORM OSCILLOSCOPE (Optimized 128 points, 0 shadowBlur) ---
        ctx.lineWidth = 1.5;
        ctx.strokeStyle = isRecording ? '#5C141A' : 'rgba(92, 20, 26, 0.85)';
        ctx.beginPath();

        const step = 4;
        const sliceWidth = (width / bufferLength) * step;
        let x = 0;

        for (let i = 0; i < bufferLength; i += step) {
          const v = isActive ? timeData[i] / 255.0 : 0.5;
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

      } else if (mode === 'spectrum') {
        // --- 2. FFT SPECTRAL BARS ---
        const barCount = 36;
        const barWidth = (width / barCount) - 2;
        const step = Math.floor(bufferLength / barCount);

        for (let i = 0; i < barCount; i++) {
          let barHeight = 2;
          if (isActive) {
            const val = freqData[i * step] || 0;
            barHeight = Math.max(4, (val / 255) * (height - 20));
          }

          const x = i * (barWidth + 2);
          const y = height - barHeight;

          ctx.fillStyle = '#5C141A';
          ctx.fillRect(x, y, barWidth, barHeight);
        }

      } else if (mode === 'phase') {
        // --- 3. HARMONIC PHASE LISSAJOUS / ORBIT (Optimized 60 samples) ---
        const centerX = width / 2;
        const centerY = height / 2;
        const radius = Math.min(width, height) * 0.38;

        ctx.lineWidth = 1.5;
        ctx.strokeStyle = '#5C141A';
        ctx.beginPath();

        const samples = 60;
        for (let i = 0; i < samples; i++) {
          const angle = (i / samples) * Math.PI * 2;
          let amp = 1;

          if (isActive) {
            const dataIdx = Math.floor((i / samples) * (bufferLength / 2));
            const freqVal = (freqData[dataIdx] || 0) / 255.0;
            amp = 0.8 + freqVal * 0.6;
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
      }

      if (isActive) {
        phaseOffset += 0.03;
      }
    };

    animationFrameId.current = requestAnimationFrame(render);

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
            PEAK: <span ref={peakRef} className="metric-val">-∞ dBFS</span>
          </span>
        </div>

        <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
          <span className="dsp-metric-pill font-mono">
            FREQ: <span ref={freqRef} className="metric-val">0 Hz</span>
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
