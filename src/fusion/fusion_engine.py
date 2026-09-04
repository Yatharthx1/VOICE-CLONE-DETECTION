from typing import Dict, Optional
import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from ..analysis.manager import ParallelAnalysisOutput
from ..detector.classifier import DeepfakePrediction
from ..verification.verifier import SpeakerVerificationResult


class FusedScore(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    raw_fused_score: float
    modality_scores: Dict[str, float] = Field(default_factory=dict)
    modality_weights: Dict[str, float] = Field(default_factory=dict)
    impersonation_penalty: float = 0.0


class ScoreFusionEngine:
    # Fusing 5 different DSP branches + 1 neural net so no single feature gets fooled

    DEFAULT_WEIGHTS = {
        "ml_detector": 0.75,
        "synthesis_artifacts": 0.08,
        "acoustic": 0.07,
        "spectral": 0.04,
        "prosody": 0.04,
        "phase": 0.02,
    }

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = weights or self.DEFAULT_WEIGHTS
        total = sum(self.weights.values())
        self.normalized_weights = {k: v / total for k, v in self.weights.items()}

    def fuse(
        self,
        analysis: ParallelAnalysisOutput,
        ml_prediction: DeepfakePrediction,
        speaker_res: Optional[SpeakerVerificationResult] = None
    ) -> FusedScore:
        # Collect the suspect scores from all branches
        scores = {
            "acoustic": float(analysis.acoustic.anomaly_score),
            "spectral": float(analysis.spectral.anomaly_score),
            "prosody": float(analysis.prosody.anomaly_score),
            "synthesis_artifacts": float(analysis.synthesis_artifacts.anomaly_score),
            "phase": float(analysis.phase.anomaly_score),
            "ml_detector": float(ml_prediction.synthetic_probability),
        }

        # Linear weighted baseline
        linear_fused = sum(scores[mod] * self.normalized_weights[mod] for mod in scores)

        # Ground truth anchoring:
        # If the trained neural classifier predicts human (< 0.25), prevent heuristic noise from pulling it up
        # If the classifier predicts AI (>= 0.70), ensure final fused score reflects strong AI
        ml_score = scores["ml_detector"]
        if ml_score < 0.25:
            fused = min(linear_fused, max(ml_score, linear_fused * 0.1))
        elif ml_score >= 0.70:
            fused = max(linear_fused, ml_score)
        else:
            fused = linear_fused

        # If they claimed to be an enrolled VIP and failed biometric matching, apply impersonation penalty
        impersonation_penalty = 0.0
        if speaker_res and speaker_res.impersonation_detected and speaker_res.claimed_speaker_id:
            impersonation_penalty = (1.0 - speaker_res.similarity_score) * 0.4
            fused = min(1.0, fused + impersonation_penalty)

        return FusedScore(
            raw_fused_score=float(round(fused, 4)),
            modality_scores={k: float(round(v, 4)) for k, v in scores.items()},
            modality_weights={k: float(round(v, 4)) for k, v in self.normalized_weights.items()},
            impersonation_penalty=float(round(impersonation_penalty, 4))
        )
