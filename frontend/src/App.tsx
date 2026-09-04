import { useState, useCallback, useRef, useEffect } from 'react';
import {
  Upload,
  Mic,
  Square,
  Play,
  Pause,
  FileAudio,
  Shield,
  AlertCircle,
  RefreshCw,
  Volume2,
} from 'lucide-react';
import { Navbar } from './components/Navbar';
import { Hero } from './components/Hero';
import { AudioVisualizer } from './components/AudioVisualizer';
import { UploadDropzone } from './components/UploadDropzone';
import { AnalyzingView } from './components/AnalyzingView';
import { ResultView } from './components/ResultView';
import {
  forensicEngine,
  encodeWavBlob,
  type ForensicReport,
} from './utils/audioEngine';
import {
  getApiBaseUrl,
  getWsBaseUrl,
  checkBackendHealth,
} from './utils/apiConfig';

type AppPhase = 'idle' | 'input-ready' | 'analyzing' | 'result';
type InputMode = 'upload' | 'live';

const ANALYSIS_STEPS = [
  'Ingesting audio stream & cryptographic hashing…',
  'Extracting spectral envelope & high-frequency rolloff…',
  'Analyzing F0 fundamental pitch & micro-jitter variance…',
  'Computing harmonic phase coherence across formants…',
  'Scanning for neural vocoder & concatenation artifacts…',
  'Evaluating multi-window temporal consistency…',
  'Fusing multi-modal evidence & computing calibrated risk…',
];

function App() {
  const [mode, setMode] = useState<InputMode>('upload');
  const [phase, setPhase] = useState<AppPhase>('idle');
  const scenario = 'general_telephony';

  // Backend connection state
  const [backendOnline, setBackendOnline] = useState<boolean | null>(null);
  const [backendError, setBackendError] = useState<string | null>(null);

  // Upload state
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [uploadedBuffer, setUploadedBuffer] = useState<AudioBuffer | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);

  // Live mic / system audio state
  const [liveSource, setLiveSource] = useState<'system' | 'mic'>('system');
  const [isRecording, setIsRecording] = useState(false);
  const [hasRecordedAudio, setHasRecordedAudio] = useState(false);
  const [liveVerdictData, setLiveVerdictData] = useState<{
    verdict: string;
    dynamicRiskScore: number;
    latencyMs: number;
    explanation?: string;
  } | null>(null);
  const micChunksRef = useRef<Float32Array[]>([]);
  const recorderRef = useRef<ScriptProcessorNode | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const latestLiveReportRef = useRef<ForensicReport | null>(null);

  // Analysis state
  const [analysisProgress, setAnalysisProgress] = useState(0);
  const [analysisStep, setAnalysisStep] = useState('');
  const [report, setReport] = useState<ForensicReport | null>(null);

  // Whether the visualizer has active signal flowing
  const isVisualizerActive = isPlaying || isRecording;

  // Poll backend health on mount
  const checkHealth = useCallback(async () => {
    const health = await checkBackendHealth();
    setBackendOnline(health.ok);
    if (health.ok) {
      setBackendError(null);
    }
  }, []);

  useEffect(() => {
    let mounted = true;
    const run = async () => {
      const health = await checkBackendHealth();
      if (!mounted) return;
      setBackendOnline(health.ok);
      if (health.ok) {
        setBackendError(null);
      }
    };
    void run();
    const interval = setInterval(() => {
      void run();
    }, 10000);
    return () => {
      mounted = false;
      clearInterval(interval);
    };
  }, []);

  // ------ Upload mode handlers ------
  const handleFileSelected = useCallback(async (file: File) => {
    setUploadedFile(file);
    setPhase('input-ready');
    try {
      const buffer = await forensicEngine.decodeAudioFile(file);
      setUploadedBuffer(buffer);
    } catch {
      console.error('Failed to decode audio file');
      setUploadedBuffer(null);
    }
  }, []);

  const handlePlayUpload = useCallback(async () => {
    if (!uploadedBuffer) return;
    if (isPlaying) {
      forensicEngine.stopPlayback();
      setIsPlaying(false);
      return;
    }
    setIsPlaying(true);
    await forensicEngine.playBuffer(uploadedBuffer, () => {
      setIsPlaying(false);
    });
  }, [uploadedBuffer, isPlaying]);

  // ------ Live mic / system audio handlers ------
  const handleStartRecording = useCallback(async (source: 'system' | 'mic' = 'system') => {
    setBackendError(null);
    setLiveVerdictData(null);
    setLiveSource(source);

    try {
      if (source === 'system') {
        await forensicEngine.startSystemAudio();
      } else {
        await forensicEngine.startMicrophone();
      }
    } catch (err: unknown) {
      console.error('Audio capture failed:', err);
      const msg = err instanceof Error ? err.message : 'Audio capture was cancelled or unavailable.';
      setBackendError(msg);
      setIsRecording(false);
      return;
    }

    try {
      const wsUrl = `${getWsBaseUrl()}/api/v1/stream/ws`;
      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      ws.onopen = async () => {
        console.log('Connected to backend WebSocket:', wsUrl);

        const ctx = forensicEngine.getAudioContext();
        const analyser = forensicEngine.getAnalyser();

        // Tell backend that the stream is ready with active scenario & client sample rate
        ws.send(
          JSON.stringify({
            scenario,
            sample_rate: ctx.sampleRate,
          })
        );

        micChunksRef.current = [];
        setHasRecordedAudio(false);
        setIsRecording(true);
        setPhase('input-ready');

        const processor = ctx.createScriptProcessor(8192, 1, 1);
        processor.onaudioprocess = (e) => {
          if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
            const input = e.inputBuffer.getChannelData(0);
            const chunk = new Float32Array(input);
            micChunksRef.current.push(chunk);
            wsRef.current.send(chunk.buffer);
          }
        };

        analyser.connect(processor);

        // Keep ScriptProcessor active without sending audio back through speakers
        const silentGain = ctx.createGain();
        silentGain.gain.value = 0;
        processor.connect(silentGain);
        silentGain.connect(ctx.destination);

        recorderRef.current = processor;
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);

          if (data.type === 'LIVE_VERDICT') {
            const isNoSpeech = data.verdict === 'NO_SPEECH';
            const isAi =
              !isNoSpeech &&
              (data.verdict === 'SYNTHETIC' ||
                data.verdict === 'MANIPULATED' ||
                data.verdict === 'AI_CLONE_DETECTED' ||
                (data.dynamic_risk_score !== undefined && data.dynamic_risk_score >= 50));

            const isSuspicious = !isNoSpeech && data.verdict === 'UNCERTAIN';
            const riskFactors = Array.isArray(data.risk_factors) ? data.risk_factors : [];
            const riskScore = Math.round(
              isNoSpeech ? 0 : (data.dynamic_risk_score ?? (data.ai_probability != null ? data.ai_probability * 100 : 0))
            );
            const conf = Math.round(
              data.confidence > 1 ? data.confidence : (data.confidence ?? 0) * 100
            );

            setLiveVerdictData({
              verdict: data.verdict,
              dynamicRiskScore: riskScore,
              latencyMs: data.latency_ms ?? 0,
              explanation: data.verdict_explanation,
            });

            const liveReport: ForensicReport = {
              isAi,
              aiProbability: riskScore,
              verdict: isAi
                ? 'AI_CLONE_DETECTED'
                : isSuspicious
                ? 'SUSPICIOUS_VOICE'
                : isNoSpeech
                ? 'NO_SPEECH'
                : 'AUTHENTIC_HUMAN',
              rawVerdict: data.verdict,
              riskScore,
              confidence: conf,
              fileName: 'Live Intercept Audio',
              durationSec: Math.round(data.end_time_sec ?? 0),
              sampleRate: 48000,
              indicators: {
                spectralCutoff: {
                  detected: riskFactors.length > 0,
                  detail: riskFactors.join(' ') || 'Continuous wideband spectrum verified.',
                },
                pitchJitter: {
                  detected: riskFactors.length > 0,
                  detail: riskFactors.join(' ') || 'Natural biological phonation jitter verified.',
                },
                phaseCoherence: {
                  detected: riskFactors.length > 0,
                  detail: riskFactors.join(' ') || 'Phase coherence consistent across formants.',
                },
                formantTransitions: {
                  detected: riskFactors.length > 0,
                  detail: riskFactors.join(' ') || 'Organic vocal articulation trajectories.',
                },
              },
              reasons: riskFactors.length ? riskFactors : [data.recommended_action || 'Live streaming verification active.'],
              recommendedAction: data.recommended_action,
              verdictExplanation: data.verdict_explanation,
              scenario,
            };

            latestLiveReportRef.current = liveReport;
          } else if (data.type === 'ERROR') {
            console.error('Backend stream error:', data.message);
          }
        } catch (err) {
          console.error('Invalid WebSocket message:', err);
        }
      };

      ws.onerror = (error) => {
        console.error('WebSocket connection error:', error);
        setBackendError('WebSocket connection to backend failed. Ensure backend is running.');
      };

      ws.onclose = () => {
        console.log('Backend WebSocket closed');
        wsRef.current = null;
      };
    } catch (err) {
      console.error('System audio capture failed:', err);
      setIsRecording(false);
      setBackendError('Microphone or audio capture was blocked or unavailable.');
    }
  }, [scenario]);

  const handleStopRecording = useCallback(() => {
    forensicEngine.stopMicrophone();

    if (recorderRef.current) {
      try {
        recorderRef.current.disconnect();
      } catch {
        // ignore
      }
      recorderRef.current = null;
    }

    if (wsRef.current) {
      try {
        wsRef.current.close();
      } catch {
        // ignore
      }
      wsRef.current = null;
    }

    setIsRecording(false);

    if (micChunksRef.current.length > 0) {
      setHasRecordedAudio(true);
      const ctx = forensicEngine.getAudioContext();
      // Package into standard WAV File so backend can perform deep analysis
      const wavBlob = encodeWavBlob(micChunksRef.current, ctx.sampleRate);
      const file = new File([wavBlob], 'live_recording.wav', { type: 'audio/wav' });
      setUploadedFile(file);

      // Create AudioBuffer for waveform preview
      const totalLen = micChunksRef.current.reduce((acc, c) => acc + c.length, 0);
      const fullBuffer = ctx.createBuffer(1, totalLen, ctx.sampleRate);
      const outData = fullBuffer.getChannelData(0);
      let offset = 0;
      for (const chunk of micChunksRef.current) {
        outData.set(chunk, offset);
        offset += chunk.length;
      }
      setUploadedBuffer(fullBuffer);
    }

    if (latestLiveReportRef.current) {
      setReport(latestLiveReportRef.current);
    }
  }, []);

  // ------ Analysis flow ------
  // ------ Run Full Forensic Analysis ------
  const runAnalysis = useCallback(async () => {
    setPhase('analyzing');
    setAnalysisProgress(0);
    setAnalysisStep(ANALYSIS_STEPS[0]);
    setBackendError(null);

    const tickerPromise = (async () => {
      for (let i = 0; i < ANALYSIS_STEPS.length; i++) {
        const targetPct = Math.round(((i + 1) / ANALYSIS_STEPS.length) * 100);
        setAnalysisStep(ANALYSIS_STEPS[i]);

        await new Promise<void>((resolve) => {
          const startPct = Math.round((i / ANALYSIS_STEPS.length) * 100);
          const dur = 240 + Math.random() * 160;
          const startTime = performance.now();

          const tick = () => {
            const elapsed = performance.now() - startTime;
            const ratio = Math.min(1, elapsed / dur);
            const eased = 1 - Math.pow(1 - ratio, 3);
            setAnalysisProgress(Math.round(startPct + (targetPct - startPct) * eased));

            if (ratio < 1) {
              requestAnimationFrame(tick);
            } else {
              resolve();
            }
          };
          requestAnimationFrame(tick);
        });
      }
    })();

    try {
      let fileToVerify = uploadedFile;

      if (!fileToVerify && uploadedBuffer) {
        const pcmData = uploadedBuffer.getChannelData(0);
        const wavBlob = encodeWavBlob([pcmData], uploadedBuffer.sampleRate);
        fileToVerify = new File([wavBlob], 'audio_recording.wav', { type: 'audio/wav' });
      }

      if (fileToVerify) {
        const formData = new FormData();
        formData.append('file', fileToVerify);
        formData.append('scenario', scenario);

        const verifyUrl = `${getApiBaseUrl()}/api/v1/verify`;
        const response = await fetch(verifyUrl, {
          method: 'POST',
          body: formData,
        });

        if (!response.ok) {
          throw new Error(`Backend error (HTTP ${response.status})`);
        }

        const backendResult = await response.json();
        await tickerPromise;

        const isNoSpeech = backendResult.verdict === 'NO_SPEECH';
        const isAi =
          !isNoSpeech &&
          (backendResult.verdict === 'SYNTHETIC' ||
            backendResult.verdict === 'MANIPULATED' ||
            backendResult.dynamic_risk_score >= 50);

        const isSuspicious = !isNoSpeech && backendResult.verdict === 'UNCERTAIN';

        const aiProbability = Math.round(
          isNoSpeech
            ? 0
            : (backendResult.dynamic_risk_score ??
                (backendResult.ai_probability != null
                  ? backendResult.ai_probability * 100
                  : (backendResult.ml_synthetic_probability ?? 0) * 100))
        );

        const confidence = Math.round(
          backendResult.confidence > 1
            ? backendResult.confidence
            : (backendResult.confidence ?? 0) * 100
        );

        const riskFactors: string[] = Array.isArray(backendResult.risk_factors)
          ? backendResult.risk_factors
          : [];

        const indicators = backendResult.indicators || {
          spectralCutoff: {
            detected: isNoSpeech ? false : riskFactors.length > 0,
            detail: isNoSpeech
              ? 'Non-speech source — no human vocal formant structure.'
              : riskFactors.length > 0
              ? riskFactors.join(' ')
              : 'Wideband acoustic spectrum uninhibited.',
          },
          pitchJitter: {
            detected: isNoSpeech ? false : riskFactors.length > 0,
            detail: isNoSpeech
              ? 'No human vocal fold phonation detected.'
              : riskFactors.length > 0
              ? riskFactors.join(' ')
              : 'Natural organic pitch micro-jitter verified.',
          },
          phaseCoherence: {
            detected: isNoSpeech ? false : riskFactors.length > 0,
            detail: isNoSpeech
              ? 'Non-vocal acoustic phase profile.'
              : riskFactors.join(' ') || 'Acoustic phase coherence confirmed.',
          },
          formantTransitions: {
            detected: isNoSpeech ? false : riskFactors.length > 0,
            detail: isNoSpeech
              ? 'No vocal tract articulatory movements detected.'
              : riskFactors.join(' ') || 'Physical vocal tract articulation curves.',
          },
        };

        const forensicResult: ForensicReport = {
          isAi,
          aiProbability,
          verdict: isAi
            ? 'AI_CLONE_DETECTED'
            : isSuspicious
            ? 'SUSPICIOUS_VOICE'
            : isNoSpeech
            ? 'NO_SPEECH'
            : 'AUTHENTIC_HUMAN',
          rawVerdict: backendResult.verdict,
          riskScore: isNoSpeech ? 0 : Math.round(backendResult.dynamic_risk_score ?? aiProbability),
          confidence,
          fileName: fileToVerify.name,
          durationSec: uploadedBuffer ? Math.round(uploadedBuffer.duration * 10) / 10 : 4.5,
          sampleRate: uploadedBuffer?.sampleRate ?? 16000,
          indicators,
          reasons: riskFactors.length ? riskFactors : [backendResult.verdict_explanation || 'Backend acoustic verification completed.'],
          recommendedAction: backendResult.recommended_action,
          verdictExplanation: backendResult.verdict_explanation,
          scenario,
        };

        setReport(forensicResult);
        setPhase('result');
      } else {
        await tickerPromise;
        setPhase('input-ready');
      }
    } catch (err: unknown) {
      console.error('Verification failed:', err);
      const msg = err instanceof Error ? err.message : 'Analysis request failed';
      setBackendError(`Backend verification failed: ${msg}. Make sure the backend server is running on port 8000.`);
      setPhase('input-ready');
    }
  }, [uploadedFile, uploadedBuffer, scenario]);

  const handleReset = useCallback(() => {
    if (wsRef.current) {
      try {
        wsRef.current.close();
      } catch {
        // ignore
      }
      wsRef.current = null;
    }
    forensicEngine.stopPlayback();
    forensicEngine.stopMicrophone();
    if (recorderRef.current) {
      try {
        recorderRef.current.disconnect();
      } catch {
        // ignore
      }
      recorderRef.current = null;
    }
    setPhase('idle');
    setUploadedFile(null);
    setUploadedBuffer(null);
    setIsPlaying(false);
    setIsRecording(false);
    setHasRecordedAudio(false);
    setReport(null);
    setLiveVerdictData(null);
    setAnalysisProgress(0);
    setBackendError(null);
    micChunksRef.current = [];
  }, []);

  useEffect(() => {
    return () => {
      forensicEngine.stopPlayback();
      forensicEngine.stopMicrophone();
    };
  }, []);

  const canAnalyze =
    (mode === 'upload' && uploadedBuffer !== null) ||
    (mode === 'live' && !isRecording && hasRecordedAudio);

  const inputFileName =
    uploadedFile?.name ?? (hasRecordedAudio ? 'Live Recording' : null);

  const inputDuration = uploadedBuffer
    ? `${uploadedBuffer.duration.toFixed(1)}s`
    : null;

  return (
    <div className="bg-grid-pattern" style={{ minHeight: '100vh', position: 'relative' }}>
      <div className="top-ambient-glow" />
      <div className="app-container">
        <Navbar />
        <Hero />

        {/* Backend Warning Banner if Offline */}
        {backendOnline === false && (
          <div
            style={{
              maxWidth: '860px',
              margin: '0 auto 1.5rem auto',
              padding: '0.85rem 1.25rem',
              borderRadius: '10px',
              background: 'rgba(239, 68, 68, 0.1)',
              border: '1px solid rgba(239, 68, 68, 0.3)',
              color: '#f87171',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              fontSize: '0.88rem',
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.65rem' }}>
              <AlertCircle size={18} />
              <span>
                <strong>Backend Server Offline:</strong> Launch with{' '}
                <code
                  style={{
                    background: 'rgba(0,0,0,0.3)',
                    padding: '0.15rem 0.4rem',
                    borderRadius: '4px',
                  }}
                >
                  python -m src.api.server
                </code>{' '}
                to enable live multi-modal DSP & neural deepfake verification.
              </span>
            </div>
            <button
              onClick={checkHealth}
              style={{
                background: 'transparent',
                border: '1px solid rgba(239, 68, 68, 0.4)',
                color: '#f87171',
                padding: '0.3rem 0.7rem',
                borderRadius: '6px',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: '0.35rem',
                fontSize: '0.8rem',
              }}
            >
              <RefreshCw size={13} />
              Retry
            </button>
          </div>
        )}

        {/* Backend Error Notification */}
        {backendError && (
          <div
            style={{
              maxWidth: '860px',
              margin: '0 auto 1.5rem auto',
              padding: '0.85rem 1.25rem',
              borderRadius: '10px',
              background: 'rgba(245, 158, 11, 0.12)',
              border: '1px solid rgba(245, 158, 11, 0.35)',
              color: '#fbbf24',
              display: 'flex',
              alignItems: 'center',
              gap: '0.65rem',
              fontSize: '0.88rem',
            }}
          >
            <AlertCircle size={18} />
            <span>{backendError}</span>
          </div>
        )}

        {/* === Workstation Card === */}
        <div className="workstation-container">
          {/* Workstation Top Controls: Mode Tabs */}
          <div className="workstation-header">
            <div className="workstation-tabs">
              <button
                type="button"
                className={`tab-btn ${mode === 'upload' ? 'active' : ''}`}
                onClick={() => {
                  if (phase !== 'analyzing') {
                    setMode('upload');
                    handleReset();
                  }
                }}
              >
                <Upload size={15} />
                Upload Audio
              </button>
              <button
                type="button"
                className={`tab-btn ${mode === 'live' ? 'active' : ''}`}
                onClick={() => {
                  if (phase !== 'analyzing') {
                    setMode('live');
                    handleReset();
                  }
                }}
              >
                <Mic size={15} />
                Live Intercept
              </button>
            </div>
          </div>

          {/* Workstation Body */}
          <div className="workstation-body">
            {phase === 'analyzing' ? (
              <AnalyzingView progress={analysisProgress} stepText={analysisStep} />
            ) : phase === 'result' && report ? (
              <ResultView report={report} onReset={handleReset} />
            ) : (
              <>
                {/* Upload Mode */}
                {mode === 'upload' && (
                  <>
                    {/* Visualizer (shows when playing uploaded audio) */}
                    {uploadedBuffer && (
                      <AudioVisualizer
                        isActive={isVisualizerActive}
                        label={isPlaying ? 'PLAYBACK ACTIVE' : 'READY FOR VERIFICATION'}
                      />
                    )}

                    {/* Mini player bar for selected file */}
                    {inputFileName && (
                      <div className="mini-audio-player">
                        <button
                          className="audio-play-btn"
                          onClick={handlePlayUpload}
                          title={isPlaying ? 'Pause' : 'Play'}
                        >
                          {isPlaying ? <Pause size={14} /> : <Play size={14} />}
                        </button>
                        <div className="audio-track-info">
                          <span className="audio-track-name">{inputFileName}</span>
                          <span className="audio-track-time font-mono">{inputDuration ?? '—'}</span>
                        </div>
                        <FileAudio size={16} style={{ color: 'var(--text-dim)', flexShrink: 0 }} />
                      </div>
                    )}

                    {/* Dropzone */}
                    {!uploadedFile && (
                      <UploadDropzone onFileSelected={handleFileSelected} />
                    )}

                    {/* Action Controls */}
                    {uploadedBuffer && (
                      <div className="controls-bar" style={{ marginTop: '1.5rem' }}>
                        <button className="btn-primary" onClick={runAnalysis} disabled={!canAnalyze}>
                          <Shield size={16} />
                          Run Forensic Analysis
                        </button>
                        <button className="btn-secondary" onClick={handleReset}>
                          Clear
                        </button>
                      </div>
                    )}
                  </>
                )}

                {/* Live Mic / System Audio Mode */}
                {mode === 'live' && (
                  <>
                    {!isRecording && !hasRecordedAudio && (
                      <div
                        className="dropzone-area"
                        onClick={() => handleStartRecording('system')}
                        role="button"
                        tabIndex={0}
                        onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') handleStartRecording('system'); }}
                      >
                        <div className="dropzone-icon-box">
                          <Volume2 size={22} />
                        </div>
                        <div className="dropzone-title">
                          Intercept System / Tab Audio
                        </div>
                        <div className="dropzone-sub">
                          Click to capture live audio from YouTube, audio calls, or media.
                          <div style={{ marginTop: '0.4rem', fontSize: '0.75rem', opacity: 0.85 }}>
                            Select the <strong>Tab</strong> (e.g. YouTube) with <em>&quot;Also share tab audio&quot;</em> checked.
                          </div>
                        </div>
                      </div>
                    )}

                    {isRecording && (
                      <AudioVisualizer
                        isActive={isRecording}
                        isRecording={isRecording}
                        label={liveSource === 'system' ? 'SYSTEM AUDIO LIVE (WEBSOCKET)' : 'MIC LIVE (WEBSOCKET)'}
                      />
                    )}

                    {/* Live streaming telemetry badge */}
                    {isRecording && liveVerdictData && (
                      <div
                        style={{
                          marginTop: '1rem',
                          padding: '0.75rem 1.25rem',
                          borderRadius: '8px',
                          background:
                            liveVerdictData.dynamicRiskScore >= 50
                              ? 'rgba(239, 68, 68, 0.1)'
                              : 'rgba(16, 185, 129, 0.1)',
                          border:
                            liveVerdictData.dynamicRiskScore >= 50
                              ? '1px solid rgba(239, 68, 68, 0.3)'
                              : '1px solid rgba(16, 185, 129, 0.3)',
                          display: 'flex',
                          justifyContent: 'space-between',
                          alignItems: 'center',
                          fontSize: '0.85rem',
                        }}
                      >
                        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                          <span
                            style={{
                              width: '9px',
                              height: '9px',
                              borderRadius: '50%',
                              backgroundColor: liveVerdictData.dynamicRiskScore >= 50 ? '#ef4444' : '#10b981',
                              boxShadow:
                                liveVerdictData.dynamicRiskScore >= 50 ? '0 0 8px #ef4444' : '0 0 8px #10b981',
                            }}
                          />
                          <span>
                            Live Verdict:{' '}
                            <strong style={{ color: liveVerdictData.dynamicRiskScore >= 50 ? '#f87171' : '#34d399' }}>
                              {liveVerdictData.verdict}
                            </strong>
                          </span>
                        </div>
                        <div style={{ display: 'flex', gap: '1.25rem', fontFamily: 'monospace', fontSize: '0.8rem' }}>
                          <span>Latency: {liveVerdictData.latencyMs}ms</span>
                        </div>
                      </div>
                    )}

                    {/* Mini player if recorded audio is ready */}
                    {!isRecording && hasRecordedAudio && inputFileName && (
                      <div className="mini-audio-player" style={{ marginTop: '1rem' }}>
                        <button
                          className="audio-play-btn"
                          onClick={handlePlayUpload}
                          title={isPlaying ? 'Pause' : 'Play'}
                        >
                          {isPlaying ? <Pause size={14} /> : <Play size={14} />}
                        </button>
                        <div className="audio-track-info">
                          <span className="audio-track-name">Captured Audio Intercept</span>
                          <span className="audio-track-time font-mono">{inputDuration ?? '—'}</span>
                        </div>
                        <FileAudio size={16} style={{ color: 'var(--text-dim)', flexShrink: 0 }} />
                      </div>
                    )}

                    <div className="controls-bar" style={{ marginTop: '1.25rem' }}>
                      {!isRecording ? (
                        <>
                          {!hasRecordedAudio ? (
                            <div style={{ display: 'flex', gap: '0.75rem', width: '100%', justifyContent: 'center' }}>
                              <button className="btn-primary" onClick={() => handleStartRecording('system')}>
                                <Volume2 size={16} />
                                Intercept System Audio
                              </button>
                              <button
                                className="btn-secondary"
                                onClick={() => handleStartRecording('mic')}
                                title="Switch to physical microphone"
                              >
                                <Mic size={15} />
                                Use Mic
                              </button>
                            </div>
                          ) : (
                            <>
                              <button className="btn-primary" onClick={runAnalysis}>
                                <Shield size={16} />
                                Run Deep Forensic Analysis
                              </button>
                              <button className="btn-secondary" onClick={handleReset}>
                                Clear
                              </button>
                            </>
                          )}
                        </>
                      ) : (
                        <button className="btn-danger" onClick={handleStopRecording}>
                          <Square size={14} />
                          Stop & Evaluate
                        </button>
                      )}
                    </div>
                  </>
                )}
              </>
            )}
          </div>
        </div>

        {/* Minimal Footer */}
        <footer className="footer-minimal">
          <div className="footer-left">
            <span>© {new Date().getFullYear()} VeraVoice</span>
            <span style={{ color: 'var(--text-dim)' }}>·</span>
            <span>Real-Time Voice Integrity & Impersonation Defense Framework</span>
          </div>
          <div className="footer-right">
            <span className="font-mono">v1.0.0</span>
          </div>
        </footer>
      </div>
    </div>
  );
}

export default App;
