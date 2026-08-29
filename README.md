# Real-Time Voice Cloning & Deepfake Detection Framework
**AICTE Cyber Security Cell — Problem ID: 26104**

---

## 🎙️ Executive Summary & Overview

An AI-powered, end-to-end voice integrity verification system designed to detect synthesized, cloned, and manipulated voices in real-time. Built specifically to mitigate deepfake threats across telephony, VoIP, contact centers, and core banking infrastructure.

The architecture strictly follows the specifications outlined in [`ARCHITECTURE.md`](./ARCHITECTURE.md) and [`PS1 PRD.md`](./PS1%20PRD.md).

---

## 🏛️ System Architecture Workflow

```
INPUT AUDIO (WAV / MP3 / M4A / MP4 / FLAC / OGG / Real-Time System Audio Stream)
  │
  ├── 1. AUDIO INGESTION (Multi-format decoding, SHA-256/MD5 forensic preservation, metadata extraction)
  │
  ├── 2. PREPROCESSING (Mono downmixing, 16/24kHz polyphase resampling, VAD segmentation, sliding chunking)
  │
  ├── PARALLEL ANALYSIS ENGINES (Executed concurrently in parallel threads):
  │     ├── Acoustic Analysis (F0 contour, Jitter: Local/RAP/PPQ5, Shimmer: Local/APQ3/APQ5/dB, LPC Formants, HNR, CPP)
  │     ├── Spectral Analysis (STFT, Mel-Spectrogram, Centroid, Spread, Skewness, Kurtosis, Flux, Rolloff, Brickwall Cutoff)
  │     ├── Prosody Analysis (Speaking rate, Syllable nuclei, nPVI / rPVI rhythm variability, Monotone pitch detection)
  │     ├── Synthesis Artifacts (Neural vocoder 2D periodic checkerboard patterns, Splicing glitch boundaries, Harmonic smearing)
  │     └── Phase Analysis (Instantaneous Frequency deviation/clustering, Modified Group Delay MGD, Phase dispersion entropy)
  │
  ├── 3. FEATURE EXTRACTION & 4. NORMALIZATION (Canonical 60-D dense feature vector + Robust Z-score baseline standardization)
  │
  ├── 5. ML DEEPFAKE DETECTOR (PyTorch Multi-Branch Residual Deep Neural Network with Layer Attribution Explainability)
  │
  ├── 6. SPEAKER VERIFICATION (128-D Acoustic Biometric Voiceprints, Genuine enrollment registry, 1-to-1 Cosine Matching)
  │
  ├── 7. SCORE FUSION & 8. CONFIDENCE CALIBRATION (Multi-modal weighted Bayesian fusion + Platt sigmoid calibration)
  │
  ├── DYNAMIC RISK SCORING (FR-02: Scenario-adaptive threshold logic for High-Value Transfer, Confidential, and Support tiers)
  │
  ├── FR-03 ALERTING & INTERVENTION (Multi-channel dispatch to Webhook, In-App, Console, and Automated Step-up MFA policies)
  │
  └── REST API & WEBSOCKET SERVICE (FastAPI server with live stream packet processing, file upload, and Swagger UI at /docs)
```

---

## 📁 Repository Directory Structure

```
PS1/
├── src/
│   ├── __init__.py
│   ├── engine.py                         # Master VoiceIntegrityEngine orchestrator
│   ├── ingestion/                        # 1. AUDIO INGESTION & 2. PREPROCESSING
│   │   ├── __init__.py
│   │   ├── config.py                     # IngestionConfig (Pydantic settings)
│   │   ├── models.py                     # IngestedAudio, ForensicRecord, AudioMetadata, VADSegment, AudioChunk
│   │   ├── forensic.py                   # SHA-256 / MD5 forensic preservation
│   │   ├── metadata_extractor.py         # FFprobe container inspection & acoustic stats
│   │   ├── decoder.py                    # Multi-format decoder (WAV, MP3, M4A, MP4, FLAC, OGG)
│   │   ├── preprocessor.py               # Mono downmixing, polyphase resampling, DC filter, normalize
│   │   ├── vad.py                        # Voice Activity Detector (energy & temporal hangover)
│   │   ├── chunker.py                    # Sliding window chunker & real-time streaming ring buffer
│   │   └── pipeline.py                   # AudioIngestionPipeline
│   ├── analysis/                         # PARALLEL ANALYSIS ENGINES
│   │   ├── __init__.py
│   │   ├── base.py                       # BaseAnalysisModule & BaseAnalysisResult
│   │   ├── acoustic/                     # ACOUSTIC ANALYSIS
│   │   │   ├── config.py, models.py, pitch.py, formants.py, voice_quality.py, analyzer.py
│   │   ├── spectral/                     # SPECTRAL ANALYSIS
│   │   │   ├── config.py, models.py, spectrogram.py, spectral_flux.py, high_frequency.py, analyzer.py
│   │   ├── prosody/                      # PROSODY ANALYSIS
│   │   │   ├── config.py, models.py, rhythm.py, intonation.py, analyzer.py
│   │   ├── synthesis_artifacts/          # SYNTHESIS ARTIFACTS ANALYSIS
│   │   │   ├── config.py, models.py, neural_vocoder.py, concatenation.py, analyzer.py
│   │   ├── phase/                        # PHASE ANALYSIS
│   │   │   ├── config.py, models.py, instantaneous_freq.py, phase_consistency.py, analyzer.py
│   │   └── manager.py                    # ParallelAnalysisManager (Concurrent execution)
│   ├── features/                         # 3. FEATURE EXTRACTION & 4. NORMALIZATION
│   │   ├── __init__.py, schema.py, extractor.py, normalizer.py
│   │   └── (Canonical 60-D dense feature vector representation)
│   ├── detector/                         # 5. ML DEEPFAKE DETECTOR
│   │   ├── __init__.py, model.py, classifier.py
│   │   └── (PyTorch ForensicAcousticDeepfakeNet multi-branch residual classifier)
│   ├── verification/                     # 6. SPEAKER VERIFICATION
│   │   ├── __init__.py, embeddings.py, database.py, verifier.py
│   │   └── (128-D Biometric voiceprint extraction & 1-to-1 matching)
│   ├── fusion/                           # 7. SCORE FUSION & 8. CONFIDENCE CALIBRATION & RISK ENGINE
│   │   ├── __init__.py, fusion_engine.py, calibrator.py, risk_engine.py
│   │   └── (Scenario-adaptive thresholds: High-Value, Confidential, Support)
│   ├── alerting/                         # FR-03 ALERTING & INTERVENTION
│   │   ├── __init__.py, notifier.py, workflow.py
│   │   └── (Multi-channel dispatcher: Webhook, In-App, Console, Event Bus)
│   └── api/                              # REST API (FASTAPI)
│       ├── __init__.py, server.py, routes.py, schemas.py
│       └── (FastAPI REST service with interactive Web UI & WebSocket streaming)
├── tests/
│   ├── conftest.py                       # Fixtures & synthetic vocal signal generators
│   ├── test_ingestion.py                 # Ingestion & Preprocessing unit tests
│   ├── test_acoustic_analysis.py         # Acoustic analysis unit tests
│   ├── test_spectral_analysis.py         # Spectral analysis unit tests
│   ├── test_prosody_analysis.py          # Prosody analysis unit tests
│   ├── test_synthesis_artifacts.py       # Synthesis artifacts unit tests
│   ├── test_phase_analysis.py            # Phase analysis unit tests
│   ├── test_features_and_ml.py           # Feature extraction & ML detector unit tests
│   ├── test_speaker_verification.py      # Speaker verification unit tests
│   ├── test_fusion_and_risk.py           # Score fusion & risk engine unit tests
│   ├── test_end_to_end_system.py         # Complete end-to-end framework verification
│   ├── test_api.py                       # REST API integration tests
│   └── run_interactive_check.py          # Interactive master CLI check tool
├── samples/                              # Multi-format generated speech samples
├── output/                               # Exported forensic archives, preprocessed audio & JSON reports
├── requirements.txt                      # Project dependencies
├── ARCHITECTURE.md                       # Architectural Blueprint
├── PS1 PRD.md                            # Product Requirements Document
└── README.md                             # Comprehensive Documentation
```

---

## 🧪 Automated Test Suite (51 Passing Tests)

Run the full automated test suite covering all modules:

```bash
python -m pytest tests/ -v
```

---

## 🛠️ Interactive Live Stream Receiver & Verification Dashboard

Run [`tests/run_interactive_check.py`](./tests/run_interactive_check.py):

### 1. Launch Live System Audio & Telephony Stream Dashboard:
```bash
python tests/run_interactive_check.py --server
```
*Open `http://localhost:8000` in your browser.*

### 2. Real-Time Live Streaming VoIP Simulation:
```bash
python tests/run_interactive_check.py --stream
```

### 3. Verify Any Received Audio File:
```bash
python tests/run_interactive_check.py --input "path/to/received_call.mp3"
```

### 4. Enroll a Genuine VIP Speaker Profile:
```bash
python tests/run_interactive_check.py --enroll "path/to/genuine_voice.wav" --speaker-id CXO_001 --speaker-name "John Doe (CEO)"
```

---

## 💻 Python SDK API Quickstart

```python
from src.engine import VoiceIntegrityEngine
from src.fusion.risk_engine import RiskScenario

# 1. Initialize engine
engine = VoiceIntegrityEngine(default_scenario=RiskScenario.HIGH_VALUE_TRANSACTION)

# 2. Perform verification
result = engine.verify_file("path/to/voice_sample.mp3", claimed_speaker_id="USER_01")

# 3. Inspect outputs
print(result.summary())
print(f"Verdict: {result.assessment.verdict.value}")
print(f"Risk Score: {result.assessment.dynamic_risk_score:.1f} / 100.0")
print(f"Action: {result.assessment.recommended_action}")
print(f"ML Deepfake Probability: {result.ml_prediction.synthetic_probability * 100:.1f}%")
```
