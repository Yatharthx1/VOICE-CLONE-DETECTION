from typing import Dict, Optional
import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from src.analysis.manager import ParallelAnalysisOutput
from src.detector.classifier import DeepfakePrediction
from src.verification.verifier import SpeakerVerificationResult


class FusedScore(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    raw_fused_score: float
    modality_scores: Dict[str, float] = Field(default_factory=dict)
    modality_weights: Dict[str, float] = Field(default_factory=dict)
    impersonation_penalty: float = 0.0


class ScoreFusionEngine:
    # Fusing 5 different DSP branches + 1 neural net so no single feature gets fooled

    DEFAULT_WEIGHTS = {
        "acoustic": 0.20,
        "spectral": 0.15,
        "prosody": 0.15,
        "synthesis_artifacts": 0.20,
        "phase": 0.10,
        "ml_detector": 0.20,
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
            "acoustic": analysis.acoustic.anomaly_score,
            "spectral": analysis.spectral.anomaly_score,
            "prosody": analysis.prosody.anomaly_score,
            "synthesis_artifacts": analysis.synthesis_artifacts.anomaly_score,
            "phase": analysis.phase.anomaly_score,
            "ml_detector": ml_prediction.synthetic_probability,
        }

        fused = sum(scores[mod] * self.normalized_weights[mod] for mod in scores)

        # If they claimed to be the CEO and failed biometric matching, slap an impersonation penalty on top
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
