"""
Modified Group Delay (MGD) and Phase Consistency Module.
"""

from typing import Tuple
import numpy as np

from .config import PhaseAnalysisConfig
from .models import GroupDelayMetrics


class PhaseConsistencyAnalyzer:
    """Computes Modified Group Delay (MGD) and phase dispersion entropy."""

    def __init__(self, config: PhaseAnalysisConfig):
        self.config = config

    def analyze(self, audio: np.ndarray, sample_rate: int) -> GroupDelayMetrics:
        """
        Compute MGD peak prominence and phase dispersion metrics.
        """
        n_fft = self.config.n_fft
        hop = self.config.hop_length
        if len(audio) < n_fft:
            return GroupDelayMetrics()

        num_frames = (len(audio) - n_fft) // hop + 1
        if num_frames < 3:
            return GroupDelayMetrics()

        # Frame signal
        shape = (num_frames, n_fft)
        strides = (audio.strides[0] * hop, audio.strides[0])
        window = np.hanning(n_fft)
        x_frames = np.lib.stride_tricks.as_strided(audio, shape=shape, strides=strides) * window

        # Time-weighted signal y[n] = n * x[n]
        n_vec = np.arange(n_fft)[None, :]
        y_frames = x_frames * n_vec

        # FFTs
        X = np.fft.rfft(x_frames, n=n_fft, axis=-1).T  # (n_freqs, num_frames)
        Y = np.fft.rfft(y_frames, n=n_fft, axis=-1).T

        X_R = np.real(X)
        X_I = np.imag(X)
        Y_R = np.real(Y)
        Y_I = np.imag(Y)

        # Numerator of Group Delay: X_R * Y_R + X_I * Y_I
        num = X_R * Y_R + X_I * Y_I
        mag = np.abs(X) + 1e-6
        denom = mag ** (2.0 * self.config.mgd_gamma)

        mgd = num / denom
        # Apply ceiling/floor to handle zeros
        mgd_mean = np.mean(mgd, axis=1)

        # Prominence of MGD peaks
        mgd_peaks = np.diff(np.sign(np.diff(mgd_mean))) < 0
        peak_prom = float(np.std(mgd_mean) / (np.mean(np.abs(mgd_mean)) + 1e-6))
        peak_prom_score = float(np.clip(peak_prom / 5.0, 0.0, 1.0))

        # Phase derivative across frequency (roughness)
        phase = np.angle(X)
        phase_unwrapped = np.unwrap(phase, axis=0)
        d2_phase = np.diff(phase_unwrapped, n=2, axis=0)
        roughness = float(np.mean(np.abs(d2_phase)))

        # Dispersion entropy of phase derivative
        d1_phase = np.diff(phase_unwrapped, n=1, axis=0).flatten()
        hist, _ = np.histogram(d1_phase, bins=30, density=True)
        hist = hist[hist > 1e-9]
        entropy = -float(np.sum(hist * np.log2(hist + 1e-12))) / 5.0  # normalize
        entropy = float(np.clip(entropy, 0.0, 1.0))

        return GroupDelayMetrics(
            mgd_peak_prominence=float(round(peak_prom_score, 3)),
            phase_dispersion_entropy=float(round(entropy, 3)),
            unwrapped_phase_roughness=float(round(roughness, 3))
        )
