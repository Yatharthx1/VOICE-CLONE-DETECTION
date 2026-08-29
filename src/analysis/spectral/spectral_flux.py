"""
Spectral Distribution and Dynamics Feature Extractor.
Extracts Centroid, Spread, Skewness, Kurtosis, Flux, Rolloff, and Flatness.
"""

from typing import Tuple
import numpy as np

from .models import SpectralMoments, SpectralDynamics


class SpectralDynamicsExtractor:
    """Computes statistical spectral moments and temporal dynamics."""

    def __init__(self, rolloff_percentile: float = 0.85):
        self.rolloff_percentile = rolloff_percentile

    def extract_moments(self, magnitude: np.ndarray, sample_rate: int) -> SpectralMoments:
        """
        Compute statistical spectral moments (Centroid, Spread, Skewness, Kurtosis).
        """
        n_freqs, n_frames = magnitude.shape
        freqs = np.linspace(0, sample_rate / 2.0, n_freqs)

        # Normalize per-frame magnitude spectrum to form probability distribution
        col_sums = np.sum(magnitude, axis=0, keepdims=True)
        col_sums = np.where(col_sums > 1e-9, col_sums, 1e-9)
        prob = magnitude / col_sums

        # 1st moment: Centroid (mean frequency)
        centroid = np.dot(freqs, prob)  # shape (n_frames,)

        # 2nd central moment: Spread / Variance
        diff = freqs[:, None] - centroid[None, :]  # shape (n_freqs, n_frames)
        spread = np.sqrt(np.sum(prob * (diff ** 2), axis=0) + 1e-12)

        # 3rd standardized moment: Skewness
        skewness = np.sum(prob * (diff ** 3), axis=0) / (spread ** 3 + 1e-12)

        # 4th standardized moment: Kurtosis
        kurtosis = np.sum(prob * (diff ** 4), axis=0) / (spread ** 4 + 1e-12)

        return SpectralMoments(
            centroid_mean_hz=float(round(np.mean(centroid), 1)),
            centroid_std_hz=float(round(np.std(centroid), 1)),
            spread_mean_hz=float(round(np.mean(spread), 1)),
            skewness_mean=float(round(np.mean(skewness), 3)),
            kurtosis_mean=float(round(np.mean(kurtosis), 3))
        )

    def extract_dynamics(self, magnitude: np.ndarray, sample_rate: int) -> SpectralDynamics:
        """
        Compute Spectral Flux, Rolloff, Flatness, and Crest Factor.
        """
        n_freqs, n_frames = magnitude.shape
        freqs = np.linspace(0, sample_rate / 2.0, n_freqs)

        # 1. Spectral Flux (frame-to-frame normalized distance)
        if n_frames > 1:
            diff_mag = np.diff(magnitude, axis=1)
            # Half-wave rectified flux
            flux = np.sqrt(np.mean(np.maximum(0, diff_mag) ** 2, axis=0))
            flux_mean = float(np.mean(flux))
            flux_std = float(np.std(flux))
        else:
            flux_mean = 0.0
            flux_std = 0.0

        # 2. Spectral Rolloff (frequency below which rolloff_percentile of energy lies)
        cumulative_energy = np.cumsum(magnitude ** 2, axis=0)
        total_energy = cumulative_energy[-1:, :]
        threshold_energy = self.rolloff_percentile * total_energy

        rolloff_freqs = []
        for frame_idx in range(n_frames):
            idx = np.where(cumulative_energy[:, frame_idx] >= threshold_energy[0, frame_idx])[0]
            if len(idx) > 0:
                rolloff_freqs.append(freqs[idx[0]])
            else:
                rolloff_freqs.append(freqs[-1])
        rolloff_mean = float(np.mean(rolloff_freqs)) if rolloff_freqs else 0.0

        # 3. Spectral Flatness (Wiener entropy = geometric mean / arithmetic mean)
        eps = 1e-10
        arithmetic_mean = np.mean(magnitude, axis=0) + eps
        geometric_mean = np.exp(np.mean(np.log(magnitude + eps), axis=0))
        flatness = geometric_mean / arithmetic_mean
        flatness_mean = float(np.mean(flatness))

        # 4. Spectral Crest Factor (peak to RMS ratio)
        peak_val = np.max(magnitude, axis=0)
        rms_val = np.sqrt(np.mean(magnitude ** 2, axis=0)) + eps
        crest = peak_val / rms_val
        crest_mean = float(np.mean(crest))

        return SpectralDynamics(
            flux_mean=float(round(flux_mean, 4)),
            flux_std=float(round(flux_std, 4)),
            rolloff_mean_hz=float(round(rolloff_mean, 1)),
            flatness_mean=float(round(flatness_mean, 4)),
            crest_factor_mean=float(round(crest_mean, 2))
        )
