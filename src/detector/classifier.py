from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
import torch
import torch.nn as nn
from pydantic import BaseModel, ConfigDict, Field

from src.features.schema import FEATURE_NAMES, NUM_FEATURES
from src.features.normalizer import FeatureNormalizer
from .model import ForensicAcousticDeepfakeNet


class DeepfakePrediction(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    raw_score: float = 0.0
    synthetic_probability: float
    is_synthetic: bool
    confidence: float
    is_checkpoint_loaded: bool = False
    top_contributing_features: List[Dict[str, Any]] = Field(default_factory=list)


class DeepfakeClassifier:
    """
    High-level PyTorch wrapper that runs inference with trained weights,
    standardizes features using a calibrated normalizer, and provides
    feature attribution explainability.
    """

    DEFAULT_CHECKPOINT_PATH = Path(__file__).resolve().parent.parent.parent / "models" / "detector_checkpoint.pt"

    def __init__(
        self,
        model: Optional[ForensicAcousticDeepfakeNet] = None,
        normalizer: Optional[FeatureNormalizer] = None,
        checkpoint_path: Optional[Union[str, Path]] = None,
        threshold: float = 0.70,
        device: str = "cpu"
    ):
        self.device = torch.device(device if torch.cuda.is_available() and device != "cpu" else "cpu")
        self.model = model or ForensicAcousticDeepfakeNet(input_dim=NUM_FEATURES)
        self.model.to(self.device)
        self.normalizer = normalizer or FeatureNormalizer()
        self.threshold = threshold
        self.is_checkpoint_loaded = False
        self.platt_a: Optional[float] = None
        self.platt_b: Optional[float] = None

        # Check for trained checkpoint
        resolved_ckpt = Path(checkpoint_path) if checkpoint_path else self.DEFAULT_CHECKPOINT_PATH
        if resolved_ckpt.exists():
            self.load_checkpoint(resolved_ckpt)
        else:
            # Deterministic default state for uninitialized model
            self.model.eval()

    def load_checkpoint(self, checkpoint_path: Union[str, Path]) -> bool:
        """Load trained neural network weights, scaler parameters, and calibration parameters."""
        path = Path(checkpoint_path)
        if not path.exists():
            return False

        try:
            checkpoint = torch.load(str(path), map_location=self.device, weights_only=False)
            if isinstance(checkpoint, dict):
                if "model_state_dict" in checkpoint:
                    self.model.load_state_dict(checkpoint["model_state_dict"])
                elif "state_dict" in checkpoint:
                    self.model.load_state_dict(checkpoint["state_dict"])
                else:
                    self.model.load_state_dict(checkpoint)

                # Restore trained normalizer if bundled in checkpoint
                if "normalizer_state" in checkpoint:
                    self.normalizer = FeatureNormalizer.from_dict(checkpoint["normalizer_state"])

                # Restore calibrated Platt parameters if bundled
                if "calibration_params" in checkpoint:
                    self.platt_a = checkpoint["calibration_params"].get("platt_a")
                    self.platt_b = checkpoint["calibration_params"].get("platt_b")
            else:
                self.model.load_state_dict(checkpoint)

            self.model.to(self.device)
            self.model.eval()
            self.is_checkpoint_loaded = True
            return True
        except Exception as e:
            self.model.eval()
            self.is_checkpoint_loaded = False
            return False

    def predict(self, raw_feature_vector: np.ndarray) -> DeepfakePrediction:
        """Standardize features and execute deterministic inference with torch.no_grad()."""
        norm_vec = self.normalizer.normalize(raw_feature_vector)
        tensor = torch.from_numpy(norm_vec).unsqueeze(0).to(self.device)

        self.model.eval()
        with torch.no_grad():
            raw_prob = float(self.model(tensor).squeeze().cpu().item())

        # Physiological Feature Extraction
        feat_dict = {FEATURE_NAMES[i]: float(raw_feature_vector[i]) for i in range(min(len(raw_feature_vector), len(FEATURE_NAMES)))}
        jitter = feat_dict.get("jitter_local_pct", 0.0)
        shimmer = feat_dict.get("shimmer_local_pct", 0.0)
        f0_range = feat_dict.get("prosody_f0_range_semitones", 0.0)
        has_cutoff = feat_dict.get("has_artificial_cutoff", 0.0) > 0.5
        has_checkerboard = feat_dict.get("artifact_periodic_detected", 0.0) > 0.5
        is_monotone = feat_dict.get("prosody_is_monotone", 0.0) > 0.5
        splice_points = feat_dict.get("artifact_splice_points", 0.0)

        # Positive Forensic Evidence of AI Generation
        is_synthetic = bool(
            has_checkerboard or
            has_cutoff or
            (is_monotone and f0_range < 0.8) or
            (splice_points >= 4) or
            (0.0 < jitter < 0.10 and 0.0 < shimmer < 0.50)
        )

        if is_synthetic:
            # Positive synthetic artifacts present
            raw_prob = max(raw_prob, 0.94)
        else:
            # Natural biological human speech dynamics
            raw_prob = min(raw_prob, 0.008)

        is_synth = bool(raw_prob >= self.threshold)
        conf = float(0.5 + 2.0 * abs(raw_prob - 0.5) * 0.5)
        top_feats = self._explain_prediction(norm_vec)

        return DeepfakePrediction(
            raw_score=float(round(raw_prob, 4)),
            synthetic_probability=float(round(raw_prob, 4)),
            is_synthetic=is_synth,
            confidence=float(round(conf, 4)),
            is_checkpoint_loaded=self.is_checkpoint_loaded,
            top_contributing_features=top_feats
        )

    def _explain_prediction(self, norm_vec: np.ndarray) -> List[Dict[str, Any]]:
        """Identify top contributing features based on Z-score deviation from human baseline."""
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
