from typing import Dict, List, Optional
import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from src.analysis.manager import ParallelAnalysisOutput
from .fusion_engine import FusedScore


class CalibratedResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    calibrated_probability: float
    confidence: float
    modality_agreement_ratio: float


class ConfidenceCalibrator:
    # Platt scaling: Turning raw heuristic scores into well-behaved probabilities

    def __init__(self, platt_a: float = 6.0, platt_b: float = -3.0):
        self.platt_a = platt_a
        self.platt_b = platt_b

    def calibrate(
        self,
        fused: FusedScore,
        analysis: ParallelAnalysisOutput,
        speech_duration_sec: float
    ) -> CalibratedResult:
        raw_score = fused.raw_fused_score

        # Classic logistic sigmoid transformation
        logit = self.platt_a * raw_score + self.platt_b
        calibrated_prob = 1.0 / (1.0 + np.exp(-logit))
        calibrated_prob = float(np.clip(calibrated_prob, 0.0, 1.0))

        # Check if the 5 analyzers agree or if they're having an argument
        mod_scores = list(fused.modality_scores.values())
        binary_decisions = [s >= 0.45 for s in mod_scores]
        majority = sum(binary_decisions) >= (len(binary_decisions) / 2.0)
        agreement_count = sum(1 for d in binary_decisions if d == majority)
        agreement_ratio = agreement_count / len(binary_decisions) if binary_decisions else 1.0

        # Short audio gets lower confidence because you can't judge a 0.2s grunt
        dur_factor = min(1.0, speech_duration_sec / 3.0)
        avg_mod_conf = np.mean([
            analysis.acoustic.confidence,
            analysis.spectral.confidence,
            analysis.prosody.confidence,
            analysis.synthesis_artifacts.confidence,
            analysis.phase.confidence,
        ])

        final_conf = float(0.4 * dur_factor + 0.3 * agreement_ratio + 0.3 * avg_mod_conf)
        final_conf = float(np.clip(final_conf, 0.1, 0.99))

        return CalibratedResult(
            calibrated_probability=float(round(calibrated_prob, 4)),
            confidence=float(round(final_conf, 4)),
            modality_agreement_ratio=float(round(agreement_ratio, 3))
        )
