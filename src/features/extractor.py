from typing import Dict, List, Optional
import numpy as np

from src.analysis.manager import ParallelAnalysisOutput
from .schema import FEATURE_NAMES, NUM_FEATURES


class FeatureExtractor:
    # Packages dictionary outputs from the 5 DSP modules into a tidy numpy vector

    def __init__(self, feature_names: Optional[List[str]] = None):
        self.feature_names = feature_names or FEATURE_NAMES

    def extract_vector(self, analysis_output: ParallelAnalysisOutput) -> np.ndarray:
        feats_dict = analysis_output.aggregated_features()
        vector = np.zeros(len(self.feature_names), dtype=np.float32)

        for i, name in enumerate(self.feature_names):
            val = feats_dict.get(name, 0.0)
            # Kill NaNs and Infinities with fire before they poison the neural net
            if val is None or np.isnan(val) or np.isinf(val):
                val = 0.0
            vector[i] = float(val)

        return vector

    def extract_dict(self, vector: np.ndarray) -> Dict[str, float]:
        return {name: float(vector[i]) for i, name in enumerate(self.feature_names) if i < len(vector)}
