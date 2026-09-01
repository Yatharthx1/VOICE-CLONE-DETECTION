"""
Neural Vocoder Checkerboard and Periodic Artifact Detector.
"""

from typing import Tuple
import numpy as np

from .config import SynthesisArtifactsConfig
from .models import NeuralVocoderArtifacts


class NeuralVocoderDetector:
    """Detects 2D spectral periodicities and harmonic smearing from neural upsampling kernels."""

    def __init__(self, config: SynthesisArtifactsConfig):
        self.config = config

    def analyze(self, audio: np.ndarray, sample_rate: int) -> NeuralVocoderArtifacts:
        """
        Extract 2D FFT periodicity and harmonic sharpness metrics.
        """
        if len(audio) < self.config.checkerboard_fft_size * 2:
            return NeuralVocoderArtifacts()

        # Compute STFT
        n_fft = self.config.checkerboard_fft_size
        hop = n_fft // 4
        window = np.hanning(n_fft)
        num_frames = (len(audio) - n_fft) // hop + 1

        if num_frames < 8:
            return NeuralVocoderArtifacts()

        # Extract spectrogram matrix
        shape = (num_frames, n_fft)
        strides = (audio.strides[0] * hop, audio.strides[0])
        frames = np.lib.stride_tricks.as_strided(audio, shape=shape, strides=strides) * window
        spec = np.abs(np.fft.rfft(frames, n=n_fft, axis=-1)).T  # shape (n_freqs, num_frames)

        # 1. 2D FFT on high frequency band (where transposed conv checkerboard artifacts manifest)
        n_freqs = spec.shape[0]
        hf_spec = spec[int(n_freqs * 0.4):, :]  # Top 60% frequency bands

        if hf_spec.shape[0] >= 16 and hf_spec.shape[1] >= 16:
            # 2D FFT
            fft2 = np.abs(np.fft.fft2(hf_spec - np.mean(hf_spec)))
            fft2_center = np.fft.fftshift(fft2)
            
            # Find energy ratio of high-spatial-frequency quadrants vs DC center
            h, w = fft2_center.shape
            ch, cw = h // 2, w // 2
            center_region = fft2_center[ch - 2: ch + 3, cw - 2: cw + 3]
            outer_corners = (
                fft2_center[:4, :4].sum() +
                fft2_center[-4:, :4].sum() +
                fft2_center[:4, -4:].sum() +
                fft2_center[-4:, -4:].sum()
            )
            total_energy = np.sum(fft2_center) + 1e-9
            checkerboard_ratio = float(outer_corners / total_energy)
        else:
            checkerboard_ratio = 0.0

        periodic_detected = bool(checkerboard_ratio > 0.08)

        # 2. Harmonic Smearing Score (evaluates spectral peak width of harmonics)
        # Average power spectrum
        avg_spec = np.mean(spec, axis=1)
        peaks = np.where((avg_spec[1:-1] > avg_spec[:-2]) & (avg_spec[1:-1] > avg_spec[2:]))[0] + 1
        
        if len(peaks) >= 3:
            # Peak to valley ratio of prominent harmonics
            peak_vals = avg_spec[peaks]
            valleys = np.where((avg_spec[1:-1] < avg_spec[:-2]) & (avg_spec[1:-1] < avg_spec[2:]))[0] + 1
            if len(valleys) >= 3:
                valley_vals = avg_spec[valleys]
                pv_ratio = np.mean(peak_vals) / (np.mean(valley_vals) + 1e-8)
                # Lower PV ratio indicates smeared/flattened harmonics (vocoder artifact)
                smearing_score = float(np.clip(1.0 - (pv_ratio / 15.0), 0.0, 1.0))
            else:
                smearing_score = 0.0
        else:
            smearing_score = 0.0

        return NeuralVocoderArtifacts(
            checkerboard_energy_ratio=float(round(checkerboard_ratio, 4)),
            periodic_artifact_detected=periodic_detected,
            harmonic_smearing_score=float(round(smearing_score, 3))
        )
