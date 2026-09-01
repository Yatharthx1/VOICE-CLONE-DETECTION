import React from 'react';
import { Shield, Scan, Activity, Layers } from 'lucide-react';

interface AnalyzingViewProps {
  progress: number; // 0-100
  stepText: string;
}

export const AnalyzingView: React.FC<AnalyzingViewProps> = ({ progress, stepText }) => {
  return (
    <div className="analyzing-container">
      <div className="scanner-beam-wrapper">
        <div className="scanner-ring" />
        <div className="scanner-ring-inner" />
        <div className="scanner-core-icon">
          <Shield size={32} />
        </div>
      </div>

      <h2 className="analyzing-headline">Acoustic Forensic Scan</h2>
      <div className="analyzing-step-text font-mono">{stepText}</div>

      <div className="analyzing-progress-track">
        <div
          className="analyzing-progress-fill"
          style={{ width: `${progress}%` }}
        />
      </div>

      <div style={{ display: 'flex', gap: '1.25rem', justifyContent: 'center', marginTop: '0.5rem' }}>
        {[
          { icon: <Scan size={13} />, label: 'Spectral Analysis', active: progress > 10 },
          { icon: <Activity size={13} />, label: 'Jitter Detection', active: progress > 35 },
          { icon: <Layers size={13} />, label: 'Phase Coherence', active: progress > 60 },
          { icon: <Shield size={13} />, label: 'Verdict Fusion', active: progress > 85 },
        ].map((step, i) => (
          <span
            key={i}
            className="dsp-metric-pill font-mono"
            style={{
              color: step.active ? 'var(--accent-cyan)' : 'var(--text-dim)',
              borderColor: step.active ? 'var(--border-cyan)' : undefined,
              transition: 'color 0.3s, border-color 0.3s',
            }}
          >
            {step.icon}
            <span style={{ marginLeft: 4 }}>{step.label}</span>
          </span>
        ))}
      </div>
    </div>
  );
};
