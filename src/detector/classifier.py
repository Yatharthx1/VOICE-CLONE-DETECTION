from typing import Any, Dict, List, Optional
import numpy as np
import torch
import torch.nn as nn
from pydantic import BaseModel, ConfigDict, Field

from src.features.schema import FEATURE_NAMES, NUM_FEATURES
from src.features.normalizer import FeatureNormalizer
from .model import ForensicAcousticDeepfakeNet


class DeepfakePrediction(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    synthetic_probability: float
    is_synthetic: bool
    confidence: float
    top_contributing_features: List[Dict[str, Any]] = Field(default_factory=list)


class DeepfakeClassifier:
    # High-level wrapper that runs inference and explains why the model got suspicious

    def __init__(
        self,
        model: Optional[ForensicAcousticDeepfakeNet] = None,
        normalizer: Optional[FeatureNormalizer] = None,
        threshold: float = 0.50,
        device: str = "cpu"
    ):
        # Default to CPU so we don't crash users running on potato laptops
        self.device = torch.device(device if torch.cuda.is_available() and device != "cpu" else "cpu")
        self.model = model or ForensicAcousticDeepfakeNet(input_dim=NUM_FEATURES)
        self.model.to(self.device)
        self.model.eval()
        self.normalizer = normalizer or FeatureNormalizer()
        self.threshold = threshold
        self._init_weights()

    def predict(self, raw_feature_vector: np.ndarray) -> DeepfakePrediction:
        # Standardize features before feeding them to the neural net
        norm_vec = self.normalizer.normalize(raw_feature_vector)
        tensor = torch.from_numpy(norm_vec).unsqueeze(0).to(self.device)

        with torch.no_grad():
            prob = float(self.model(tensor).squeeze().cpu().item())

        is_synth = bool(prob >= self.threshold)
        # Distance from the 0.5 coinflip fence gives us our confidence
        conf = float(0.5 + 2.0 * abs(prob - 0.5) * 0.5)
        top_feats = self._explain_prediction(norm_vec)

        return DeepfakePrediction(
            synthetic_probability=float(round(prob, 4)),
            is_synthetic=is_synth,
            confidence=float(round(conf, 4)),
            top_contributing_features=top_feats
        )

    def _explain_prediction(self, norm_vec: np.ndarray) -> List[Dict[str, Any]]:
        # Pick the top 5 features with wildest Z-scores to explain the verdict
        deviations = np.abs(norm_vec)
        top_indices = np.argsort(deviations)[::-1][:5]
        top_feats = []
        for idx in top_indices:
            if idx < len(FEATURE_NAMES):
                top_feats.append({
                    "feature_name": FEATURE_NAMES[idx],
                    "z_score": float(round(norm_vec[idx], 2)),
                    "impact": "Synthetic indicator" if (norm_vec[idx] > 1.5 or norm_vec[idx] < -1.5) else "Within normal baseline"
                })
        return top_feats

    def _init_weights(self):
        # Orthogonal init because bad random weights ruin good sleep
        torch.manual_seed(42)
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if "weight" in name and param.dim() >= 2:
                    nn.init.orthogonal_(param, gain=0.8)
                elif "bias" in name:
                    nn.init.constant_(param, 0.0)
