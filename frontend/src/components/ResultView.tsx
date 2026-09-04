import {
  ShieldAlert,
  ShieldCheck,
  RotateCcw,
  Download,
  Info,
  Activity,
  Layers,
  Waves,
  CircuitBoard,
  AlertTriangle,
  Music,
} from 'lucide-react';
import type { ForensicReport } from '../utils/audioEngine';

interface ResultViewProps {
  report: ForensicReport;
  onReset: () => void;
}

export const ResultView = ({ report, onReset }: ResultViewProps) => {
  const isNoSpeech = report.verdict === 'NO_SPEECH' || report.rawVerdict === 'NO_SPEECH';
  const isAi = !isNoSpeech && report.isAi;
  const isSuspicious = !isNoSpeech && (report.verdict === 'SUSPICIOUS_VOICE' || report.rawVerdict === 'UNCERTAIN');
  const verdictClass = isAi
    ? 'ai-detected'
    : isSuspicious
    ? 'suspicious-detected'
    : isNoSpeech
    ? 'nospeech-detected'
    : 'human-verified';
  const prob = report.aiProbability;

  const meterColor = isAi
    ? `linear-gradient(90deg, #b91c1c, #c2410c)`
    : isSuspicious
    ? `linear-gradient(90deg, #d97706, #f59e0b)`
    : isNoSpeech
    ? `linear-gradient(90deg, #475569, #64748b)`
    : `linear-gradient(90deg, #15803d, #22c55e)`;

  return (
    <div className="result-container">
      {/* Verdict Header Card */}
      <div className={`verdict-header-card ${verdictClass}`}>
        <div className="verdict-left-box">
          <div className="verdict-icon-badge">
            {isAi ? (
              <ShieldAlert size={26} />
            ) : isSuspicious ? (
              <AlertTriangle size={26} />
            ) : isNoSpeech ? (
              <Music size={26} />
            ) : (
              <ShieldCheck size={26} />
            )}
          </div>
          <div>
            <div className="verdict-score-headline font-mono">
              {isNoSpeech
                ? 'Non-Speech / Music Detected'
                : `${prob}% ${isAi ? 'AI Generated (Synthetic)' : isSuspicious ? 'Suspicious / Inconclusive' : 'Human Authentic'}`}
            </div>
            <div className="verdict-sublabel">
              {report.verdictExplanation ||
                (isNoSpeech
                  ? 'Audio contains music, instruments, or non-vocal audio rather than human speech'
                  : isAi
                  ? 'Synthetic vocoder artifacts & neural TTS patterns detected'
                  : 'Organic vocal characteristics verified — no synthetic artifacts')}
            </div>
            {report.recommendedAction && (
              <div
                style={{
                  marginTop: '0.5rem',
                  fontSize: '0.78rem',
                  color: 'var(--text-muted)',
                  fontFamily: 'monospace',
                }}
              >
                Action: <span style={{ color: isAi ? '#f87171' : isNoSpeech ? '#94a3b8' : '#6ee7b7' }}>{report.recommendedAction}</span>
              </div>
            )}
          </div>
        </div>

        <div className="verdict-meta-badges">
          <span className="meta-badge font-mono">
            {report.durationSec}s @ {(report.sampleRate / 1000).toFixed(1)}kHz
          </span>
          <span className="meta-badge font-mono">
            Confidence: {report.confidence}%
          </span>
        </div>
      </div>

      {/* Confidence Meter */}
      <div className="confidence-meter-block">
        <div className="meter-header">
          <span style={{ fontWeight: 600 }}>AI Synthesis Probability</span>
          <span className="font-mono" style={{ color: isAi ? 'var(--alert-red)' : 'var(--accent-emerald)' }}>
            {prob}%
          </span>
        </div>
        <div className="meter-track">
          <div
            className="meter-fill-bar"
            style={{ width: `${prob}%`, background: meterColor }}
          />
        </div>
        <div className="meter-markers font-mono">
          <span>0% Authentic</span>
          <span>50% Threshold</span>
          <span>100% Synthetic</span>
        </div>
      </div>

      {/* Why? Forensic Explanation */}
      <div className="why-breakdown-card">
        <div className="why-header">
          <Info size={16} style={{ color: 'var(--accent-cyan)' }} />
          <span>Why this verdict?</span>
        </div>

        <div className="why-grid">
          {report.indicators.spectralCutoff.detected !== undefined && (
            <div className="why-item">
              <div className="why-item-icon"><Waves size={16} /></div>
              <div className="why-item-content">
                <div className="why-item-title">Spectral Cutoff Analysis</div>
                <div className="why-item-desc">{report.indicators.spectralCutoff.detail}</div>
              </div>
            </div>
          )}
          <div className="why-item">
            <div className="why-item-icon"><Activity size={16} /></div>
            <div className="why-item-content">
              <div className="why-item-title">Pitch Micro-Jitter</div>
              <div className="why-item-desc">{report.indicators.pitchJitter.detail}</div>
            </div>
          </div>
          <div className="why-item">
            <div className="why-item-icon"><Layers size={16} /></div>
            <div className="why-item-content">
              <div className="why-item-title">Phase Coherence</div>
              <div className="why-item-desc">{report.indicators.phaseCoherence.detail}</div>
            </div>
          </div>
          <div className="why-item">
            <div className="why-item-icon"><CircuitBoard size={16} /></div>
            <div className="why-item-content">
              <div className="why-item-title">Formant Transitions</div>
              <div className="why-item-desc">{report.indicators.formantTransitions.detail}</div>
            </div>
          </div>
        </div>
      </div>

      {/* Actions */}
      <div className="result-actions-bar">
        <button className="btn-primary" onClick={onReset}>
          <RotateCcw size={16} />
          Analyze Another
        </button>
        <button className="btn-secondary" onClick={() => {
          const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' });
          const url = URL.createObjectURL(blob);
          const a = document.createElement('a');
          a.href = url;
          a.download = `veravoice-report-${Date.now()}.json`;
          a.click();
          URL.revokeObjectURL(url);
        }}>
          <Download size={14} />
          Export Report
        </button>
      </div>
    </div>
  );
};
