from typing import Dict, List, Optional
import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from src.analysis.manager import ParallelAnalysisOutput
from .fusion_engine import FusedScore


class CalibratedResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    raw_score: float = 0.0
    calibrated_probability: float
    confidence: float
    modality_agreement_ratio: float
    window_consistency: float = 1.0
    verdict_category: str = "Ambiguous"


class ConfidenceCalibrator:
    """
    Calibrates raw acoustic/detector heuristic scores into statistically sound posterior
    probabilities, and evaluates evidence reliability (confidence) independently from probability.
    """

    def __init__(
        self,
        platt_a: Optional[float] = None,
        platt_b: Optional[float] = None,
        center: float = 0.65
    ):
        # Default empirical parameters ensuring:
        # Human (raw < 0.50) -> low AI probability
        # Ambiguous (raw 0.55-0.75) -> review / uncertain band
        # Strong AI (raw > 0.80) -> high AI probability
        if platt_a is not None and platt_b is not None:
            loaded_a = float(platt_a)
            loaded_b = float(platt_b)
            self.platt_a = float(np.clip(loaded_a if loaded_a > 0 else 8.0, 4.0, 8.0))
            midpoint = -loaded_b / loaded_a if loaded_a > 0 else center
            self.platt_b = loaded_b if 0.60 <= midpoint <= 0.75 else (-self.platt_a * center)
        else:
            self.platt_a = 8.0
            self.platt_b = -self.platt_a * center

    def calibrate(
        self,
        fused: FusedScore,
        analysis: ParallelAnalysisOutput,
        speech_duration_sec: float,
        window_consistency: float = 1.0,
        window_scores: Optional[List[float]] = None
    ) -> CalibratedResult:
        raw_score = fused.raw_fused_score

        # 1. Platt Sigmoid Calibration
        logit = self.platt_a * raw_score + self.platt_b
        # Numerical stability clip
        logit_clipped = np.clip(logit, -20.0, 20.0)
        calibrated_prob = 1.0 / (1.0 + np.exp(-logit_clipped))
        calibrated_prob = float(np.clip(calibrated_prob, 0.0, 1.0))

        # 2. Modality Agreement Analysis
        mod_scores = list(fused.modality_scores.values())
        if mod_scores:
            binary_decisions = [s >= 0.50 for s in mod_scores]
            majority = sum(binary_decisions) >= (len(binary_decisions) / 2.0)
            agreement_count = sum(1 for d in binary_decisions if d == majority)
            agreement_ratio = agreement_count / len(binary_decisions)
        else:
            agreement_ratio = 1.0

        # 3. Confidence Calculation (Durability, Agreement, Consistency, and Quality)
        dur_factor = min(1.0, max(0.1, speech_duration_sec / 3.0))
        avg_mod_conf = np.mean([
            analysis.acoustic.confidence,
            analysis.spectral.confidence,
            analysis.prosody.confidence,
            analysis.synthesis_artifacts.confidence,
            analysis.phase.confidence,
        ]) if analysis else 0.8

        w_cons = float(np.clip(window_consistency, 0.0, 1.0))

        # Weighted combination of independent reliability factors
        final_conf = float(
            0.25 * dur_factor +
            0.25 * agreement_ratio +
            0.25 * avg_mod_conf +
            0.25 * w_cons
        )

        # Penalize confidence if window scores are highly conflicting
        if window_scores and len(window_scores) > 1:
            score_range = max(window_scores) - min(window_scores)
            if score_range > 0.40:
                final_conf *= max(0.5, 1.0 - (score_range - 0.40))

        final_conf = float(np.clip(final_conf, 0.05, 0.99))

        # 4. Categorize Target Probability Tier
        if calibrated_prob < 0.20:
            category = "Clearly Human"
        elif calibrated_prob < 0.40:
            category = "Mostly Human"
        elif calibrated_prob < 0.60:
            category = "Ambiguous / Inconclusive"
        elif calibrated_prob < 0.80:
            category = "Likely AI"
        elif calibrated_prob < 0.95:
            category = "Strong AI Evidence"
        else:
            category = "Extremely Strong AI Evidence"

        return CalibratedResult(
            raw_score=float(round(raw_score, 4)),
            calibrated_probability=float(round(calibrated_prob, 4)),
            confidence=float(round(final_conf, 4)),
            modality_agreement_ratio=float(round(agreement_ratio, 3)),
            window_consistency=float(round(w_cons, 3)),
            verdict_category=category
        )
