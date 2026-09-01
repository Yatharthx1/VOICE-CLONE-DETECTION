"""
Tests for Phase Analysis Subsystem.
"""

import numpy as np
import pytest

from src.analysis.phase import (
    PhaseAnalyzer,
    PhaseAnalysisConfig,
    InstantaneousFrequencyTracker,
    PhaseConsistencyAnalyzer,
    PhaseAnalysisResult
)
from src.analysis.spectral.spectrogram import SpectrogramExtractor


def test_instantaneous_frequency_tracker():
    """Verify Instantaneous Frequency tracking on clean harmonic signal."""
    sr = 16000
    t = np.linspace(0, 1.0, 16000, endpoint=False)
    sig = (0.5 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)

    spec_ext = SpectrogramExtractor()
    _, complex_stft = spec_ext.compute_stft(sig)

    tracker = InstantaneousFrequencyTracker(config=PhaseAnalysisConfig())
    res = tracker.analyze(complex_stft, sr)

    assert res.if_harmonic_clustering_score > 0.0
    assert res.if_mean_deviation >= 0.0


def test_phase_consistency_mgd():
    """Verify Modified Group Delay calculation."""
    sr = 16000
    t = np.linspace(0, 1.0, 16000, endpoint=False)
    sig = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    analyzer = PhaseConsistencyAnalyzer(config=PhaseAnalysisConfig())
    res = analyzer.analyze(sig, sr)

    assert 0.0 <= res.mgd_peak_prominence <= 1.0
    assert 0.0 <= res.phase_dispersion_entropy <= 1.0


def test_phase_analyzer_end_to_end(speech_wav):
    """Verify complete PhaseAnalyzer output."""
    import soundfile as sf
    audio, sr = sf.read(str(speech_wav), dtype="float32")
    analyzer = PhaseAnalyzer()
    res = analyzer.analyze(audio, sr)

    assert isinstance(res, PhaseAnalysisResult)
    assert res.module_name == "phase_analysis"
    assert 0.0 <= res.anomaly_score <= 1.0
    assert "phase_if_clustering" in res.phase_features
    assert "phase_mgd_peak_prominence" in res.phase_features
