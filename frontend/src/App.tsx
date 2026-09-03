import React, { useState, useCallback, useRef, useEffect } from 'react';
import {
  Upload,
  Mic,
  Square,
  Play,
  Pause,
  FileAudio,
  Shield,
} from 'lucide-react';
import { Navbar } from './components/Navbar';
import { Hero } from './components/Hero';
import { AudioVisualizer } from './components/AudioVisualizer';
import { UploadDropzone } from './components/UploadDropzone';
import { AnalyzingView } from './components/AnalyzingView';
import { ResultView } from './components/ResultView';
import {
  forensicEngine,
  PRESET_SAMPLES,
  type ForensicReport,
  type PresetSample,
} from './utils/audioEngine';

type AppPhase = 'idle' | 'input-ready' | 'analyzing' | 'result';
type InputMode = 'upload' | 'live';

const ANALYSIS_STEPS = [
  'Ingesting audio stream…',
  'Extracting spectral envelope features…',
  'Running F0 pitch jitter detection…',
  'Computing harmonic phase coherence…',
  'Analyzing formant transition curves…',
  'Evaluating neural vocoder signatures…',
  'Fusing risk signals & computing verdict…',
];

function App() {
  const [mode, setMode] = useState<InputMode>('upload');
  const [phase, setPhase] = useState<AppPhase>('idle');

  // Upload state
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [uploadedBuffer, setUploadedBuffer] = useState<AudioBuffer | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);

  // Live mic state
  const [isRecording, setIsRecording] = useState(false);
  const [hasRecordedAudio, setHasRecordedAudio] = useState(false);
  const micChunksRef = useRef<Float32Array[]>([]);
  const recorderRef = useRef<ScriptProcessorNode | null>(null);
  const wsRef = useRef<WebSocket | null>(null);

  // Preset state
  const [selectedPreset, setSelectedPreset] = useState<PresetSample | null>(null);

  // Analysis state
  const [analysisProgress, setAnalysisProgress] = useState(0);
  const [analysisStep, setAnalysisStep] = useState('');
  const [report, setReport] = useState<ForensicReport | null>(null);

  // Whether the visualizer has active signal flowing
  const isVisualizerActive = isPlaying || isRecording;

  // ------ Upload mode handlers ------
  const handleFileSelected = useCallback(async (file: File) => {
    setUploadedFile(file);
    setSelectedPreset(null);
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

  // ------ Preset handlers ------
  const handlePresetSelect = useCallback((preset: PresetSample) => {
    setSelectedPreset(preset);
    setUploadedFile(null);
    setUploadedBuffer(null);
    setPhase('input-ready');
  }, []);

  const handlePlayPreset = useCallback(async () => {
    if (!selectedPreset) return;
    if (isPlaying) {
      forensicEngine.stopPlayback();
      setIsPlaying(false);
      return;
    }
    const buffer = forensicEngine.createPresetAudioBuffer(selectedPreset);
    setIsPlaying(true);
    await forensicEngine.playBuffer(buffer, () => {
      setIsPlaying(false);
    });
  }, [selectedPreset, isPlaying]);

  // ------ Live mic handlers ------
  const handleStartRecording = useCallback(async () => {
    try {
      // Connect to the FastAPI real-time WebSocket
      const ws = new WebSocket(
        'ws://127.0.0.1:8000/api/v1/stream/ws'
      );

      wsRef.current = ws;

      ws.onopen = async () => {
        console.log('Connected to backend WebSocket');

        // Tell backend that the stream is ready
        ws.send(
          JSON.stringify({
            scenario: 'general_telephony',
          })
        );

        await forensicEngine.startMicrophone();

        micChunksRef.current = [];
        setHasRecordedAudio(false);
        setIsRecording(true);
        setPhase('input-ready');

        const ctx = forensicEngine.getAudioContext();
        const analyser = forensicEngine.getAnalyser();

        const processor = ctx.createScriptProcessor(
          4096,
          1,
          1
        );

        processor.onaudioprocess = (e) => {
          if (
            wsRef.current &&
            wsRef.current.readyState === WebSocket.OPEN
          ) {
            const input = e.inputBuffer.getChannelData(0);

            // Keep a copy locally
            micChunksRef.current.push(
              new Float32Array(input)
            );

            // Send Float32 PCM directly to FastAPI
            wsRef.current.send(
              input.slice().buffer
            );
          }
        };

        analyser.connect(processor);

        // Keep ScriptProcessor active without sending audio
        // back through the speakers.
        const silentGain = ctx.createGain();
        silentGain.gain.value = 0;

        processor.connect(silentGain);
        silentGain.connect(ctx.destination);

        recorderRef.current = processor;
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);

          console.log('Backend WebSocket message:', data);

          if (data.type === 'CONFIG_ACK') {
            console.log('Backend stream ready');
          }

          if (data.type === 'LIVE_VERDICT') {
            console.log('LIVE VERDICT:', data);

            const isAi =
              data.verdict === 'AI_CLONE_DETECTED';

            const riskFactors = Array.isArray(data.risk_factors)
              ? data.risk_factors
              : [];

            const liveReport: ForensicReport = {
              isAi,
              aiProbability: Math.round(
                data.dynamic_risk_score ?? 0
              ),
              verdict: isAi
                ? 'AI_CLONE_DETECTED'
                : 'AUTHENTIC_HUMAN',
              riskScore: Math.round(
                data.dynamic_risk_score ?? 0
              ),
              confidence: Math.round(
                data.confidence ?? 0
              ),
              fileName: 'Live System Audio',
              durationSec: 0,
              sampleRate: 48000,

              indicators: {
                spectralCutoff: {
                  detected: riskFactors.length > 0,
                  detail:
                    riskFactors.join(' ') ||
                    'No spectral risk factor reported.'
                },
                pitchJitter: {
                  detected: riskFactors.length > 0,
                  detail:
                    riskFactors.join(' ') ||
                    'No pitch-jitter risk factor reported.'
                },
                phaseCoherence: {
                  detected: riskFactors.length > 0,
                  detail:
                    riskFactors.join(' ') ||
                    'No phase-coherence risk factor reported.'
                },
                formantTransitions: {
                  detected: riskFactors.length > 0,
                  detail:
                    riskFactors.join(' ') ||
                    'No formant-transition risk factor reported.'
                }
              },

              reasons: riskFactors.length
                ? riskFactors
                : [data.recommended_action || 'Live analysis completed.']
            };

            setReport(liveReport);
          }

          if (data.type === 'ERROR') {
            console.error(
              'Backend stream error:',
              data.message
            );
          }
        } catch (err) {
          console.error(
            'Invalid WebSocket message:',
            err
          );
        }
      };

      ws.onerror = (error) => {
        console.error(
          'WebSocket connection error:',
          error
        );
      };

      ws.onclose = () => {
        console.log('Backend WebSocket closed');
        wsRef.current = null;
      };

    } catch (err) {
      console.error(
        'System audio capture failed:',
        err
      );

      setIsRecording(false);
    }
  }, []);

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
    setHasRecordedAudio(
      micChunksRef.current.length > 0
    );
  }, []);

  // ------ Analysis flow ------
  const runAnalysis = useCallback(async () => {
    setPhase('analyzing');
    setAnalysisProgress(0);
    setAnalysisStep(ANALYSIS_STEPS[0]);

    // Simulate progressive forensic analysis stages
    for (let i = 0; i < ANALYSIS_STEPS.length; i++) {
      const targetPct = Math.round(((i + 1) / ANALYSIS_STEPS.length) * 100);
      setAnalysisStep(ANALYSIS_STEPS[i]);

      await new Promise<void>((resolve) => {
        const startPct = Math.round((i / ANALYSIS_STEPS.length) * 100);
        const dur = 350 + Math.random() * 250;
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

    // Generate the forensic report
    let forensicResult: ForensicReport;

    if (selectedPreset) {
      // Use hardcoded preset results for deterministic demos
      forensicResult = {
        isAi: selectedPreset.isAi,
        aiProbability: selectedPreset.aiProbability,
        verdict: selectedPreset.isAi ? 'AI_CLONE_DETECTED' : 'AUTHENTIC_HUMAN',
        riskScore: selectedPreset.isAi ? selectedPreset.aiProbability : 100 - selectedPreset.aiProbability,
        confidence: 95 + Math.floor(Math.random() * 4),
        fileName: selectedPreset.name,
        durationSec: selectedPreset.duration,
        sampleRate: 48000,
        indicators: {
          spectralCutoff: {
            detected: selectedPreset.isAi,
            detail: selectedPreset.isAi
              ? 'Cutoff observed at ~7.8kHz (Neural Vocoder)'
              : 'Wideband acoustic spectrum uninhibited'
          },
          pitchJitter: {
            detected: selectedPreset.isAi,
            detail: selectedPreset.isAi
              ? 'Unusually static pitch variance (0.12%)'
              : 'Organic biological human jitter (0.86%)'
          },
          phaseCoherence: {
            detected: !selectedPreset.isAi,
            detail: selectedPreset.isAi
              ? 'Phase discontinuities across phonemes'
              : 'Smooth physical vocal tract coherence'
          },
          formantTransitions: {
            detected: selectedPreset.isAi,
            detail: selectedPreset.isAi
              ? 'Mathematical linear interpolation detected'
              : 'Physical muscular articulation curves'
          }
        },
        reasons: selectedPreset.whyReasons,
      };
    } else if (uploadedFile) {
      const formData = new FormData();

      formData.append('file', uploadedFile);
      formData.append('scenario', 'general_telephony');

      const response = await fetch(
        'http://127.0.0.1:8000/api/v1/verify',
        {
          method: 'POST',
          body: formData,
        }
      );

      if (!response.ok) {
        throw new Error(`Backend returned HTTP ${response.status}`);
      }

      const backendResult = await response.json();

      const aiProbability = Math.round(
        (backendResult.ml_synthetic_probability ?? 0) * 100
      );

      const isAi =
        backendResult.verdict === 'AI_CLONE_DETECTED' ||
        aiProbability >= 50;

      const riskFactors: string[] = Array.isArray(backendResult.risk_factors)
        ? backendResult.risk_factors
        : [];

      forensicResult = {
        isAi,
        aiProbability,
        verdict: isAi
          ? 'AI_CLONE_DETECTED'
          : 'AUTHENTIC_HUMAN',
        riskScore: Math.round(
          backendResult.dynamic_risk_score ?? 0
        ),
        confidence: Math.round(
          backendResult.confidence ?? 0
        ),
        fileName: uploadedFile.name,
        durationSec: uploadedBuffer?.duration ?? 0,
        sampleRate: uploadedBuffer?.sampleRate ?? 48000,

        indicators: {
          spectralCutoff: {
            detected: riskFactors.length > 0,
            detail:
              riskFactors.length > 0
                ? riskFactors.join(' ')
                : 'No specific spectral risk factor reported by the backend.',
          },

          pitchJitter: {
            detected: riskFactors.length > 0,
            detail:
              riskFactors.length > 0
                ? riskFactors.join(' ')
                : 'No specific pitch-jitter risk factor reported by the backend.',
          },

          phaseCoherence: {
            detected: riskFactors.length > 0,
            detail:
              riskFactors.length > 0
                ? riskFactors.join(' ')
                : 'No specific phase-coherence risk factor reported by the backend.',
          },

          formantTransitions: {
            detected: riskFactors.length > 0,
            detail:
              riskFactors.length > 0
                ? riskFactors.join(' ')
                : 'No specific formant-transition risk factor reported by the backend.',
          },
        },

        reasons:
          riskFactors.length > 0
            ? riskFactors
            : ['Analysis completed by the backend.'],
      };
    } else if (micChunksRef.current.length > 0) {
      // Concatenate mic chunks into a single buffer
      const ctx = forensicEngine.getAudioContext();
      const totalLen = micChunksRef.current.reduce((acc, chunk) => acc + chunk.length, 0);
      const fullBuffer = ctx.createBuffer(1, totalLen, ctx.sampleRate);
      const output = fullBuffer.getChannelData(0);
      let offset = 0;
      for (const chunk of micChunksRef.current) {
        output.set(chunk, offset);
        offset += chunk.length;
      }
      forensicResult = forensicEngine.analyzeAudio(fullBuffer, 'Live Microphone Input');
    } else {
      // Fallback mock
      forensicResult = {
        isAi: false,
        aiProbability: 8,
        verdict: 'AUTHENTIC_HUMAN',
        riskScore: 8,
        confidence: 94,
        fileName: 'Unknown Input',
        durationSec: 0,
        sampleRate: 48000,
        indicators: {
          spectralCutoff: { detected: false, detail: 'No analysis data' },
          pitchJitter: { detected: false, detail: 'No analysis data' },
          phaseCoherence: { detected: true, detail: 'No analysis data' },
          formantTransitions: { detected: false, detail: 'No analysis data' },
        },
        reasons: ['Insufficient audio data for high-confidence analysis.'],
      };
    }

    setReport(forensicResult);
    setPhase('result');
  }, [selectedPreset, uploadedBuffer, uploadedFile]);

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
      try { recorderRef.current.disconnect(); } catch { /* ignore */ }
      recorderRef.current = null;
    }
    setPhase('idle');
    setUploadedFile(null);
    setUploadedBuffer(null);
    setSelectedPreset(null);
    setIsPlaying(false);
    setIsRecording(false);
    setHasRecordedAudio(false);
    setReport(null);
    setAnalysisProgress(0);
    micChunksRef.current = [];
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      forensicEngine.stopPlayback();
      forensicEngine.stopMicrophone();
    };
  }, []);

  // Determine if analyze button should be enabled
  const canAnalyze =
    (mode === 'upload' && (uploadedBuffer !== null || selectedPreset !== null)) ||
    (mode === 'live' && !isRecording && hasRecordedAudio);

  // Derive upload mini-player info
  const inputFileName = uploadedFile?.name ?? selectedPreset?.name ?? null;
  const inputDuration = uploadedBuffer
    ? `${uploadedBuffer.duration.toFixed(1)}s`
    : selectedPreset
      ? `${selectedPreset.duration.toFixed(1)}s`
      : null;

  return (
    <div className="bg-grid-pattern" style={{ minHeight: '100vh', position: 'relative' }}>
      <div className="top-ambient-glow" />
      <div className="app-container">
        <Navbar />
        <Hero />

        {/* === Workstation Card === */}
        <div className="workstation-container">
          {/* Mode Tabs */}
          <div className="workstation-tabs">
            <button
              type="button"
              className={`tab-btn ${mode === 'upload' ? 'active' : ''}`}
              onClick={() => { if (phase !== 'analyzing') { setMode('upload'); handleReset(); } }}
            >
              <Upload size={15} />
              Upload Audio
              <span className="tab-indicator-badge">File</span>
            </button>
            <button
              type="button"
              className={`tab-btn ${mode === 'live' ? 'active' : ''}`}
              onClick={() => { if (phase !== 'analyzing') { setMode('live'); handleReset(); } }}
            >
              <Mic size={15} />
              Intercept System Audio
              <span className="tab-indicator-badge">System</span>
            </button>
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
                    {/* Visualizer (shows when playing uploaded or preset audio) */}
                    {(uploadedBuffer || selectedPreset) && (
                      <AudioVisualizer
                        isActive={isPlaying}
                        label={isPlaying ? 'PLAYBACK ACTIVE' : 'READY'}
                      />
                    )}

                    {/* Mini player bar for selected file/preset */}
                    {inputFileName && (
                      <div className="mini-audio-player">
                        <button
                          className="audio-play-btn"
                          onClick={uploadedBuffer ? handlePlayUpload : handlePlayPreset}
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
                    {!uploadedFile && !selectedPreset && (
                      <UploadDropzone onFileSelected={handleFileSelected} />
                    )}



                    {/* Analyze button */}
                    {(uploadedBuffer || selectedPreset) && (
                      <div className="controls-bar" style={{ marginTop: '1.25rem' }}>
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

                {/* Live Mic Mode */}
                {mode === 'live' && (
                  <>
                    <AudioVisualizer
                      isActive={isRecording}
                      isRecording={isRecording}
                      label={isRecording ? 'SYSTEM AUDIO LIVE' : 'AWAITING INPUT'}
                    />

                    <div className="controls-bar" style={{ marginTop: '1.25rem' }}>
                      {!isRecording ? (
                        <>
                          <button className="btn-primary" onClick={handleStartRecording}>
                            <Mic size={16} />
                            Start Recording
                          </button>
                          {report && !isRecording && (
                            <button
                              className="btn-primary"
                              onClick={() => setPhase('result')}
                            >
                              <Shield size={16} />
                              View Result
                            </button>
                          )}
                        </>
                      ) : (
                        <button className="btn-danger" onClick={handleStopRecording}>
                          <Square size={14} />
                          Stop Recording
                        </button>
                      )}
                    </div>

                    {!isRecording && !hasRecordedAudio && (
                      <p style={{
                        textAlign: 'center',
                        color: 'var(--text-muted)',
                        fontSize: '0.85rem',
                        marginTop: '1rem',
                        lineHeight: 1.6,
                      }}>
                        Click <strong>Start Recording</strong> to capture system audio.
                        <br />
                        Play the audio for 2–5 seconds, then stop to analyze.
                      </p>
                    )}
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
            <span>Acoustic Forensics Engine</span>
          </div>
          <div className="footer-right">
            <span className="font-mono">v1.4.0</span>
          </div>
        </footer>
      </div>
    </div>
  );
}

export default App;
