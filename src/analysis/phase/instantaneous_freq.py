"""
Instantaneous Frequency (IF) Distribution and Harmonic Phase Tracking.
"""

from typing import Tuple
import numpy as np

from .config import PhaseAnalysisConfig
from .models import InstantaneousFrequencyMetrics


class InstantaneousFrequencyTracker:
    """Computes Instantaneous Frequency trajectories and harmonic grid dispersion."""

    def __init__(self, config: PhaseAnalysisConfig):
        self.config = config

    def analyze(self, complex_stft: np.ndarray, sample_rate: int) -> InstantaneousFrequencyMetrics:
        """
        Analyze unwrapped phase temporal derivatives.
        complex_stft shape: (n_freqs, n_frames)
        """
        n_freqs, n_frames = complex_stft.shape
        if n_frames < 3:
            return InstantaneousFrequencyMetrics()

        # Phase angle in radians
        phase = np.angle(complex_stft)

        # Phase derivative along time (unwrapped difference)
        d_phase = np.diff(np.unwrap(phase, axis=1), axis=1)

        # Nominal bin frequencies in radians/sample
        bin_rad = np.linspace(0, np.pi, n_freqs)[:, None]

        # Instantaneous Frequency deviation from center frequency
        inst_freq_rad = d_phase / (2.0 * np.pi)
        
        # Deviation
        deviation = np.abs(inst_freq_rad - (bin_rad / (2.0 * np.pi)))
        mean_dev = float(np.mean(deviation))
        var_dev = float(np.var(deviation))

        # Harmonic clustering: evaluate how sharply IF clusters around dominant peaks
        mag = np.abs(complex_stft[:, :-1])
        top_bins_mask = mag > np.percentile(mag, 80)
        
        if np.any(top_bins_mask):
            top_deviations = deviation[top_bins_mask]
            # Smooth inverse logistic clustering score in (0, 1]
            clustering_score = float(1.0 / (1.0 + np.mean(top_deviations) * 2.0))
        else:
            clustering_score = 0.5

        return InstantaneousFrequencyMetrics(
            if_mean_deviation=float(round(mean_dev, 4)),
            if_variance=float(round(var_dev, 5)),
            if_harmonic_clustering_score=float(round(clustering_score, 3))
        )
