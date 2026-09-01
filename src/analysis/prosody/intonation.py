"""
Macro-Prosodic Pitch Intonation and Monotone Speech Detector.
"""

from typing import List
import numpy as np

from .config import ProsodyAnalysisConfig
from .models import IntonationMetrics


class IntonationAnalyzer:
    """Evaluates macro-intonation curves, pitch dynamism, and monotonic speech traits."""

    def __init__(self, config: ProsodyAnalysisConfig):
        self.config = config

    def analyze(self, f0_contour: List[float], hop_sec: float = 0.010) -> IntonationMetrics:
        """
        Analyze pitch intonation contours, semitone range, and inflection turns.
        """
        voiced = [f for f in f0_contour if f > 0.0]
        if len(voiced) < 5:
            return IntonationMetrics()

        voiced_arr = np.array(voiced)

        # 1. Semitone conversion relative to 100 Hz
        semitones = 12.0 * np.log2(voiced_arr / 100.0)
        p5 = np.percentile(semitones, 5)
        p95 = np.percentile(semitones, 95)
        f0_range_st = float(max(0.0, p95 - p5))

        # 2. Pitch slope (gradient)
        diff_f0 = np.diff(voiced_arr) / hop_sec  # Hz per second
        slope_mean = float(np.mean(diff_f0))
        slope_var = float(np.var(diff_f0))

        # 3. Monotone check
        f0_std = float(np.std(voiced_arr))
        is_monotone = bool(f0_std < self.config.monotone_f0_std_threshold_hz or f0_range_st < 2.0)

        # 4. Direction changes (inflection turns)
        signs = np.sign(diff_f0)
        # Filter near-zero slopes
        signs = signs[np.abs(diff_f0) > 10.0]
        if len(signs) > 1:
            turns = int(np.sum(np.diff(signs) != 0))
        else:
            turns = 0

        return IntonationMetrics(
            f0_range_semitones=float(round(f0_range_st, 2)),
            pitch_slope_mean=float(round(slope_mean, 2)),
            pitch_slope_variance=float(round(slope_var, 2)),
            is_monotone=is_monotone,
            pitch_direction_changes=turns
        )
