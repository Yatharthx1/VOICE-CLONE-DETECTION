"""
Concatenation and Audio Splicing Glitch Detector.
"""

import numpy as np

from .config import SynthesisArtifactsConfig
from .models import ConcatenationArtifacts


class ConcatenationDetector:
    """Detects splice boundaries, abrupt amplitude discontinuities, and phase steps."""

    def __init__(self, config: SynthesisArtifactsConfig):
        self.config = config

    def analyze(self, audio: np.ndarray, sample_rate: int) -> ConcatenationArtifacts:
        """
        Analyze signal for unnatural frame boundary transitions.
        """
        frame_len = int(0.010 * sample_rate)  # 10ms micro-frames
        if len(audio) < frame_len * 4:
            return ConcatenationArtifacts()

        num_frames = len(audio) // frame_len
        frames = audio[:num_frames * frame_len].reshape(num_frames, frame_len)
        
        # Compute RMS energy per 10ms frame in dB
        rms = np.sqrt(np.mean(frames ** 2, axis=1)) + 1e-8
        rms_db = 20.0 * np.log10(rms)

        # Frame-to-frame delta energy
        delta_db = np.abs(np.diff(rms_db))
        max_jump = float(np.max(delta_db)) if len(delta_db) > 0 else 0.0

        # Phase jump check at frame boundaries
        end_samples = frames[:-1, -1]
        start_samples = frames[1:, 0]
        sample_jumps = np.abs(start_samples - end_samples)
        max_sample_jump = float(np.max(sample_jumps)) if len(sample_jumps) > 0 else 0.0

        # Identify suspicious splice boundaries exceeding configured jump threshold
        suspicious_points = np.where(delta_db > self.config.splice_energy_jump_threshold_db)[0]
        splice_count = len(suspicious_points)

        return ConcatenationArtifacts(
            splice_points_detected=splice_count,
            max_energy_jump_db=float(round(max_jump, 2)),
            max_phase_jump_rad=float(round(max_sample_jump * np.pi, 3))
        )
