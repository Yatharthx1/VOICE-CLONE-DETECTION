"""
Unit and Functional Tests for the Acoustic Analysis Subsystem.
Tests pitch (F0) tracking, Jitter/Shimmer perturbation, LPC formants,
voice quality (HNR, CPP), and anomaly scoring.
"""

import numpy as np
import pytest
from scipy import signal

from src.ingestion import AudioIngestionPipeline, IngestionConfig
from src.analysis.acoustic import (
    AcousticAnalyzer,
    AcousticAnalysisConfig,
    PitchAnalyzer,
    FormantAnalyzer,
    VoiceQualityAnalyzer,
    AcousticAnalysisResult,
)


# ============================================================================
# 1. Synthetic Audio Test Helpers
# ============================================================================

def make_sine(freq: float, duration: float = 1.0, sr: int = 16000) -> np.ndarray:
    """Generate a clean single-frequency sine wave."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return (0.6 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def make_synthetic_vowel(f0: float = 130.0, f1: float = 700.0, f2: float = 1200.0, f3: float = 2500.0, dur: float = 1.5, sr: int = 16000) -> np.ndarray:
    """
    Generate synthetic voiced vowel sound by filtering a glottal pulse train
    through formant resonators (second-order bandpass filters).
    """
    n_samples = int(dur * sr)
    t = np.linspace(0, dur, n_samples, endpoint=False)
    
    # Glottal pulse train with slight jitter
    pulse_train = np.zeros(n_samples, dtype=np.float32)
    period = sr / f0
    idx = 0.0
    while idx < n_samples:
        pulse_train[int(idx)] = 1.0
        # Add slight natural pitch variation (+- 1 sample)
        jitter_samp = np.random.uniform(-0.5, 0.5)
        idx += period + jitter_samp

    # Formant resonators (IIR bandpass filters)
    def resonator(freq, bw):
        r = np.exp(-np.pi * bw / sr)
        theta = 2 * np.pi * freq / sr
        b = [1 - r]
        a = [1, -2 * r * np.cos(theta), r * r]
        return signal.lfilter(b, a, pulse_train)

    vowel = resonator(f1, 80) * 1.0 + resonator(f2, 110) * 0.7 + resonator(f3, 150) * 0.4
    # Normalize
    vowel = vowel / (np.max(np.abs(vowel)) + 1e-6) * 0.8
    return vowel.astype(np.float32)


# ============================================================================
# 2. Pitch & Perturbation Tests
# ============================================================================

def test_pitch_estimation_pure_tone():
    """Verify F0 tracking accurately estimates the pitch of a 220 Hz pure tone."""
    sr = 16000
    tone = make_sine(freq=220.0, duration=1.0, sr=sr)
    analyzer = PitchAnalyzer()
    res = analyzer.analyze(tone, sr)

    assert pytest.approx(res.mean_f0_hz, abs=3.0) == 220.0
    assert res.voiced_fraction > 0.85
    assert len(res.f0_contour) > 0


def test_pitch_estimation_vocal_harmonic():
    """Verify F0 tracking on a harmonic vocal signal with fundamental 130 Hz."""
    sr = 16000
    vowel = make_synthetic_vowel(f0=130.0, dur=1.5, sr=sr)
    analyzer = PitchAnalyzer()
    res = analyzer.analyze(vowel, sr)

    assert pytest.approx(res.mean_f0_hz, abs=5.0) == 130.0
    assert res.voiced_fraction > 0.80
    assert res.jitter_local_pct >= 0.0
    assert res.shimmer_local_pct >= 0.0


# ============================================================================
# 3. Formant Analysis Tests (LPC)
# ============================================================================

def test_formant_extraction_vowel():
    """Verify LPC formant tracking discovers resonance formants near F1=700Hz and F2=1200Hz."""
    sr = 16000
    vowel = make_synthetic_vowel(f0=130.0, f1=700.0, f2=1200.0, f3=2500.0, dur=2.0, sr=sr)
    analyzer = FormantAnalyzer()
    res = analyzer.analyze(vowel, sr)

    assert res.f1_mean_hz is not None
    assert res.f2_mean_hz is not None
    # Check that estimated formants match expected resonance zones
    assert 550.0 <= res.f1_mean_hz <= 850.0
    assert 1000.0 <= res.f2_mean_hz <= 1400.0
    assert res.vocal_tract_length_cm is not None
    assert 10.0 <= res.vocal_tract_length_cm <= 22.0


# ============================================================================
# 4. Voice Quality Tests (HNR & CPP)
# ============================================================================

def test_voice_quality_harmonic_vs_noise():
    """Verify HNR and CPP are substantially higher for harmonic voice than white noise."""
    sr = 16000
    vowel = make_synthetic_vowel(f0=150.0, dur=1.0, sr=sr)
    noise = np.random.normal(0, 0.2, int(1.0 * sr)).astype(np.float32)

    analyzer = VoiceQualityAnalyzer()
    res_vowel = analyzer.analyze(vowel, sr)
    res_noise = analyzer.analyze(noise, sr)

    # Harmonic vowel should exhibit clear harmonicity
    assert res_vowel.hnr_mean_db > res_noise.hnr_mean_db
    assert res_vowel.cpp_mean_db > 0.0


# ============================================================================
# 5. Acoustic Analyzer Engine Integration Tests
# ============================================================================

def test_acoustic_analyzer_complete_pipeline(speech_wav):
    """Test running AcousticAnalyzer on an IngestedAudio instance from the ingestion pipeline."""
    pipeline = AudioIngestionPipeline()
    ingested = pipeline.process_file(speech_wav)

    analyzer = AcousticAnalyzer()
    result = analyzer.analyze_ingested(ingested)

    assert isinstance(result, AcousticAnalysisResult)
    assert result.module_name == "acoustic_analysis"
    assert 0.0 <= result.anomaly_score <= 1.0
    assert 0.0 <= result.confidence <= 1.0
    assert result.pitch.mean_f0_hz > 0.0
    assert "pitch_mean_f0" in result.acoustic_features
    assert "jitter_local_pct" in result.acoustic_features
    assert "formant_f1_mean" in result.acoustic_features
    assert "hnr_mean_db" in result.acoustic_features

    # Check summary
    summary = result.summary()
    assert "=== Acoustic Analysis Summary" in summary
    assert "Pitch (F0)" in summary


def test_acoustic_analyzer_chunk_processing(speech_wav):
    """Test parallel/sliding window chunk acoustic analysis."""
    pipeline = AudioIngestionPipeline(config=IngestionConfig(chunk_window_sec=2.0, chunk_hop_sec=1.0))
    ingested = pipeline.process_file(speech_wav)

    analyzer = AcousticAnalyzer()
    chunk_results = analyzer.analyze_chunks(ingested.chunks)

    assert len(chunk_results) == len(ingested.chunks)
    for res in chunk_results:
        assert isinstance(res, AcousticAnalysisResult)
        assert 0.0 <= res.anomaly_score <= 1.0


def test_synthetic_voice_indicator_detection():
    """Verify that perfectly synthetic periodic sine wave triggers low micro-jitter/shimmer indicators."""
    sr = 16000
    pure_tone = make_sine(freq=200.0, duration=2.0, sr=sr)
    analyzer = AcousticAnalyzer()
    res = analyzer.analyze(pure_tone, sr)

    # Pure sine wave has near 0% jitter/shimmer (robotic/vocoder artifact)
    assert res.pitch.jitter_local_pct < 0.2
    assert len(res.synthetic_indicators) > 0


def test_acoustic_analyzer_silence_edge_case():
    """Verify acoustic analyzer handles pure silence safely without exceptions."""
    sr = 16000
    silence = np.zeros(sr * 2, dtype=np.float32)
    analyzer = AcousticAnalyzer()
    res = analyzer.analyze(silence, sr)

    assert res.anomaly_score >= 0.0
    assert res.pitch.mean_f0_hz == 0.0
    assert res.pitch.voiced_fraction == 0.0
