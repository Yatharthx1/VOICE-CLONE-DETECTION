"""
Tests for Prosody Analysis Subsystem.
"""

import numpy as np
import pytest

from src.analysis.prosody import (
    ProsodyAnalyzer,
    ProsodyAnalysisConfig,
    RhythmAnalyzer,
    IntonationAnalyzer,
    ProsodyAnalysisResult
)


def test_intonation_monotone_detection():
    """Verify detection of flat robotic pitch intonation."""
    cfg = ProsodyAnalysisConfig()
    intonation_analyzer = IntonationAnalyzer(config=cfg)

    # Flat monotone pitch contour: strictly constant 150 Hz
    flat_f0 = [150.0] * 50
    res_flat = intonation_analyzer.analyze(flat_f0)
    assert res_flat.is_monotone is True
    assert res_flat.f0_range_semitones < 1.0

    # Expressive dynamic pitch contour
    t = np.linspace(0, 2.0, 50)
    dynamic_f0 = list(150.0 + 30.0 * np.sin(2 * np.pi * 1.5 * t))
    res_dyn = intonation_analyzer.analyze(dynamic_f0)
    assert res_dyn.is_monotone is False
    assert res_dyn.f0_range_semitones > 3.0


def test_rhythm_syllable_and_npvi(speech_wav):
    """Verify syllable counting and rhythm variability extraction."""
    import soundfile as sf
    audio, sr = sf.read(str(speech_wav), dtype="float32")
    cfg = ProsodyAnalysisConfig()
    rhythm_analyzer = RhythmAnalyzer(config=cfg)
    
    rhythm = rhythm_analyzer.analyze(audio, sr, speech_duration_sec=3.0)
    assert rhythm.syllable_count >= 2
    assert rhythm.speaking_rate_sps > 0.0
    assert rhythm.npvi >= 0.0


def test_prosody_analyzer_end_to_end(speech_wav):
    """Verify complete ProsodyAnalyzer integration."""
    import soundfile as sf
    audio, sr = sf.read(str(speech_wav), dtype="float32")
    analyzer = ProsodyAnalyzer()
    res = analyzer.analyze(audio, sr)

    assert isinstance(res, ProsodyAnalysisResult)
    assert res.module_name == "prosody_analysis"
    assert 0.0 <= res.anomaly_score <= 1.0
    assert "prosody_speaking_rate" in res.prosody_features
    assert "prosody_is_monotone" in res.prosody_features
