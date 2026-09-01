"""
Tests for Score Fusion, Confidence Calibration, Dynamic Risk Scoring, and Alerting.
"""

import numpy as np
import pytest

from src.analysis.manager import ParallelAnalysisManager
from src.detector.classifier import DeepfakePrediction
from src.fusion.fusion_engine import ScoreFusionEngine, FusedScore
from src.fusion.calibrator import ConfidenceCalibrator, CalibratedResult
from src.fusion.risk_engine import (
    RiskScoringEngine,
    RiskAssessment,
    RiskScenario,
    Verdict,
    RiskLevel
)
from src.alerting.notifier import AlertDispatcher
from src.alerting.workflow import InterventionWorkflow


def test_score_fusion_and_calibration(speech_wav):
    """Verify multi-modal score fusion and Platt calibration."""
    import soundfile as sf
    audio, sr = sf.read(str(speech_wav), dtype="float32")

    manager = ParallelAnalysisManager()
    analysis_out = manager.analyze_audio(audio, sr)

    fake_pred = DeepfakePrediction(
        synthetic_probability=0.20,
        is_synthetic=False,
        confidence=0.80,
        top_contributing_features=[]
    )

    fusion_engine = ScoreFusionEngine()
    fused = fusion_engine.fuse(analysis_out, fake_pred)

    assert isinstance(fused, FusedScore)
    assert 0.0 <= fused.raw_fused_score <= 1.0
    assert len(fused.modality_scores) == 6

    calibrator = ConfidenceCalibrator()
    calibrated = calibrator.calibrate(fused, analysis_out, speech_duration_sec=3.0)

    assert isinstance(calibrated, CalibratedResult)
    assert 0.0 <= calibrated.calibrated_probability <= 1.0
    assert 0.0 <= calibrated.confidence <= 1.0


def test_risk_scenarios_and_threshold_adaptation(speech_wav):
    """Verify scenario-adaptive risk thresholds and automated alerts."""
    import soundfile as sf
    audio, sr = sf.read(str(speech_wav), dtype="float32")

    manager = ParallelAnalysisManager()
    analysis_out = manager.analyze_audio(audio, sr)

    fusion_engine = ScoreFusionEngine()
    calibrator = ConfidenceCalibrator()

    # Moderate score (0.42) - should trigger HIGH RISK in High-Value Transfer scenario
    moderate_pred = DeepfakePrediction(
        synthetic_probability=0.42,
        is_synthetic=False,
        confidence=0.75,
        top_contributing_features=[]
    )
    fused = fusion_engine.fuse(analysis_out, moderate_pred)
    calibrated = calibrator.calibrate(fused, analysis_out, speech_duration_sec=3.0)

    risk_engine_bank = RiskScoringEngine(scenario=RiskScenario.HIGH_VALUE_TRANSACTION)
    assessment_bank = risk_engine_bank.assess_risk(fused, calibrated, analysis_out)

    risk_engine_support = RiskScoringEngine(scenario=RiskScenario.STANDARD_SUPPORT)
    assessment_support = risk_engine_support.assess_risk(fused, calibrated, analysis_out)

    # Banking scenario should be more sensitive / strict
    assert assessment_bank.scenario == RiskScenario.HIGH_VALUE_TRANSACTION
    assert assessment_support.scenario == RiskScenario.STANDARD_SUPPORT

    # Test automated intervention workflow
    dispatcher = AlertDispatcher()
    workflow = InterventionWorkflow(dispatcher=dispatcher)
    alert = workflow.process_assessment(assessment_bank, session_id="test_session_123")
    assert alert is not None
    assert alert.session_id == "test_session_123"
