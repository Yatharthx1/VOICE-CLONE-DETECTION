"""
High-Frequency Cutoff and Synthetic Brickwall Filter Detector.
Neural TTS vocoders and older speech synthesis engines frequently exhibit unnatural
steep spectral cutoffs (e.g. sharp drops above 3.8 kHz, 6.0 kHz, or 7.2 kHz).
"""

from typing import Optional, Tuple
import numpy as np

from .models import HighFrequencyAnalysis


class HighFrequencyCutoffDetector:
    """Detects brickwall filters and high-frequency energy voids."""

    def __init__(self, cutoff_threshold_db: float = -40.0):
        self.cutoff_threshold_db = cutoff_threshold_db

    def analyze(self, magnitude: np.ndarray, sample_rate: int) -> HighFrequencyAnalysis:
        """
        Detect artificial spectral steep roll-offs / cutoffs.
        """
        n_freqs, n_frames = magnitude.shape
        freqs = np.linspace(0, sample_rate / 2.0, n_freqs)

        # Average energy spectrum in dB
        avg_power = np.mean(magnitude ** 2, axis=1) + 1e-12
        max_power = np.max(avg_power)
        power_db = 10.0 * np.log10(avg_power / max_power)

        # Cumulative energy distribution across frequency bins
        cum_energy = np.cumsum(avg_power)
        total_energy = cum_energy[-1]
        cum_ratio = cum_energy / total_energy if total_energy > 0 else np.zeros_like(cum_energy)

        # HF Energy Ratio (energy > 4000 Hz / total energy)
        hf_mask = freqs >= 4000.0
        hf_energy = np.sum(avg_power[hf_mask]) if np.any(hf_mask) else 0.0
        hf_ratio = float(hf_energy / total_energy) if total_energy > 0 else 0.0

        has_cutoff = False
        detected_cutoff_hz = None

        # Condition 1: 99.8% of energy is concentrated below (Nyquist - 1200 Hz)
        # e.g., in a 16kHz audio (Nyquist 8kHz), if 99.8% energy is below 4000Hz or 6800Hz
        nyquist = sample_rate / 2.0
        cutoff_candidates = np.where((cum_ratio >= 0.998) & (freqs <= nyquist - 1000.0))[0]
        if len(cutoff_candidates) > 0 and freqs[cutoff_candidates[0]] >= 1000.0:
            has_cutoff = True
            detected_cutoff_hz = float(round(freqs[cutoff_candidates[0]], 1))

        # Condition 2: Steep gradient drop
        if not has_cutoff:
            start_bin = int((1500.0 / nyquist) * n_freqs)
            for i in range(start_bin, n_freqs - 5):
                if power_db[i] < self.cutoff_threshold_db and np.all(power_db[i:] < self.cutoff_threshold_db + 5.0):
                    prev_idx = max(0, i - 10)
                    gradient = power_db[i] - power_db[prev_idx]
                    if gradient < -20.0:
                        has_cutoff = True
                        detected_cutoff_hz = float(round(freqs[i], 1))
                        break

        return HighFrequencyAnalysis(
            has_artificial_cutoff=has_cutoff,
            cutoff_frequency_hz=detected_cutoff_hz,
            hf_energy_ratio=float(round(hf_ratio, 4))
        )
