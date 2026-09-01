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
        nyquist = sample_rate / 2.0

        # Detect steep brickwall filter cliff followed by dead stopband floor
        start_bin = int((1400.0 / nyquist) * n_freqs)
        for i in range(start_bin, n_freqs - 6):
            if power_db[i] < -42.0 and np.all(power_db[i:] < -35.0):
                prev_idx = max(0, i - 12)
                gradient = power_db[i] - power_db[prev_idx]
                stopband_floor = np.mean(power_db[i:])
                if gradient < -25.0 and stopband_floor < -45.0:
                    has_cutoff = True
                    detected_cutoff_hz = float(round(freqs[i], 1))
                    break

        return HighFrequencyAnalysis(
            has_artificial_cutoff=has_cutoff,
            cutoff_frequency_hz=detected_cutoff_hz,
            hf_energy_ratio=float(round(hf_ratio, 4))
        )
