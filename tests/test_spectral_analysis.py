"""
Tests for Spectral Analysis Subsystem.
"""

import numpy as np
import pytest

from src.analysis.spectral import (
    SpectralAnalyzer,
    SpectralAnalysisConfig,
    SpectrogramExtractor,
    SpectralDynamicsExtractor,
    HighFrequencyCutoffDetector,
    SpectralAnalysisResult
)


def test_spectrogram_stft_and_mel():
    """Verify STFT and Mel-spectrogram computation shapes."""
    sr = 16000
    audio = np.random.normal(0, 0.1, 16000).astype(np.float32)
    extractor = SpectrogramExtractor(n_fft=1024, hop_length=256, n_mels=80)
    
    mag, complex_stft = extractor.compute_stft(audio)
    assert mag.shape[0] == 513
    assert mag.shape[1] > 0
    assert complex_stft.shape == mag.shape

    mel = extractor.compute_mel_spectrogram(mag, sr)
    assert mel.shape[0] == 80
    assert mel.shape[1] == mag.shape[1]


def test_spectral_dynamics_and_moments():
    """Verify spectral moments and dynamics calculations."""
    sr = 16000
    # 1000 Hz sine wave has spectral centroid near 1000 Hz
    t = np.linspace(0, 1.0, 16000, endpoint=False)
    sig = np.sin(2 * np.pi * 1000 * t).astype(np.float32)
    
    spec_ext = SpectrogramExtractor()
    mag, _ = spec_ext.compute_stft(sig)

    dynamics_ext = SpectralDynamicsExtractor()
    moments = dynamics_ext.extract_moments(mag, sr)
    dynamics = dynamics_ext.extract_dynamics(mag, sr)

    assert pytest.approx(moments.centroid_mean_hz, abs=100.0) == 1000.0
    assert dynamics.rolloff_mean_hz >= 1000.0
    assert dynamics.flatness_mean < 0.2


def test_high_frequency_cutoff_detection():
    """Verify detection of brickwall filter cutoffs."""
    sr = 16000
    # Signal with steep cutoff above 3500 Hz
    t = np.linspace(0, 1.0, 16000, endpoint=False)
    sig = np.sin(2 * np.pi * 500 * t) + np.sin(2 * np.pi * 1500 * t)
    
    spec_ext = SpectrogramExtractor()
    mag, _ = spec_ext.compute_stft(sig.astype(np.float32))

    hf_detector = HighFrequencyCutoffDetector()
    res = hf_detector.analyze(mag, sr)
    assert res.has_artificial_cutoff is True
    assert res.cutoff_frequency_hz is not None
    assert res.cutoff_frequency_hz <= 4000.0


def test_spectral_analyzer_end_to_end(speech_wav):
    """Verify complete SpectralAnalyzer output."""
    import soundfile as sf
    audio, sr = sf.read(str(speech_wav), dtype="float32")
    analyzer = SpectralAnalyzer()
    res = analyzer.analyze(audio, sr)

    assert isinstance(res, SpectralAnalysisResult)
    assert res.module_name == "spectral_analysis"
    assert 0.0 <= res.anomaly_score <= 1.0
    assert "spectral_centroid_mean" in res.spectral_features
    assert "spectral_flux_mean" in res.spectral_features
