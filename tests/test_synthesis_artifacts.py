"""
Tests for Synthesis Artifacts Subsystem.
"""

import numpy as np
import pytest

from src.analysis.synthesis_artifacts import (
    SynthesisArtifactsAnalyzer,
    SynthesisArtifactsConfig,
    NeuralVocoderDetector,
    ConcatenationDetector,
    SynthesisArtifactsResult
)


def test_splice_glitch_detection():
    """Verify detection of sudden audio concatenation splice discontinuities."""
    sr = 16000
    cfg = SynthesisArtifactsConfig(splice_energy_jump_threshold_db=15.0)
    detector = ConcatenationDetector(config=cfg)

    # Audio with artificial hard splice jump (0.05 gain jumping instantly to 0.95 gain)
    p1 = 0.05 * np.sin(2 * np.pi * 300 * np.linspace(0, 0.5, 8000, endpoint=False))
    p2 = 0.95 * np.sin(2 * np.pi * 800 * np.linspace(0, 0.5, 8000, endpoint=False))
    spliced = np.concatenate([p1, p2]).astype(np.float32)

    res = detector.analyze(spliced, sr)
    assert res.splice_points_detected >= 1
    assert res.max_energy_jump_db > 15.0


def test_neural_vocoder_detector():
    """Verify neural vocoder periodic pattern extractor."""
    sr = 16000
    cfg = SynthesisArtifactsConfig()
    detector = NeuralVocoderDetector(config=cfg)

    # Standard clean audio
    t = np.linspace(0, 1.0, 16000, endpoint=False)
    clean = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    res = detector.analyze(clean, sr)
    assert isinstance(res.checkerboard_energy_ratio, float)
    assert isinstance(res.harmonic_smearing_score, float)


def test_synthesis_artifacts_analyzer_end_to_end(speech_wav):
    """Verify complete SynthesisArtifactsAnalyzer integration."""
    import soundfile as sf
    audio, sr = sf.read(str(speech_wav), dtype="float32")
    analyzer = SynthesisArtifactsAnalyzer()
    res = analyzer.analyze(audio, sr)

    assert isinstance(res, SynthesisArtifactsResult)
    assert res.module_name == "synthesis_artifacts"
    assert 0.0 <= res.anomaly_score <= 1.0
    assert "artifact_checkerboard_ratio" in res.artifact_features
    assert "artifact_splice_points" in res.artifact_features
