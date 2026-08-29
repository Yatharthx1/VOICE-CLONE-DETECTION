from typing import Optional
import numpy as np

from .schema import FEATURE_NAMES, NUM_FEATURES


class FeatureNormalizer:
    # Standardizes raw acoustic numbers so a 2000Hz centroid doesn't overwhelm a 0.05 jitter

    def __init__(
        self,
        means: Optional[np.ndarray] = None,
        stds: Optional[np.ndarray] = None
    ):
        if means is not None and stds is not None:
            self.means = means.astype(np.float32)
            self.stds = np.maximum(stds.astype(np.float32), 1e-4)
        else:
            self.means, self.stds = self._initialize_baseline_distributions()

    def normalize(self, vector: np.ndarray) -> np.ndarray:
        if len(vector) != len(self.means):
            vec = np.zeros(len(self.means), dtype=np.float32)
            min_len = min(len(vector), len(self.means))
            vec[:min_len] = vector[:min_len]
        else:
            vec = vector.astype(np.float32)

        z = (vec - self.means) / self.stds
        # Hard clip at +/- 5 sigma so rogue microphone pops don't send loss to the moon
        return np.clip(z, -5.0, 5.0).astype(np.float32)

    def denormalize(self, z_vector: np.ndarray) -> np.ndarray:
        return (z_vector * self.stds + self.means).astype(np.float32)

    @staticmethod
    def _initialize_baseline_distributions() -> tuple[np.ndarray, np.ndarray]:
        # Empirical mean and std deviations from human speech benchmarks
        means = np.zeros(NUM_FEATURES, dtype=np.float32)
        stds = np.ones(NUM_FEATURES, dtype=np.float32)

        baseline_map = {
            "pitch_mean_f0": (150.0, 45.0),
            "pitch_std_f0": (20.0, 12.0),
            "pitch_min_f0": (90.0, 30.0),
            "pitch_max_f0": (240.0, 60.0),
            "pitch_voiced_fraction": (0.65, 0.20),
            "jitter_local_pct": (0.75, 0.40),
            "jitter_rap_pct": (0.35, 0.20),
            "jitter_ppq5_pct": (0.40, 0.20),
            "shimmer_local_pct": (2.50, 1.20),
            "shimmer_apq3_pct": (1.20, 0.60),
            "shimmer_apq5_pct": (1.50, 0.70),
            "shimmer_db": (0.25, 0.15),
            "formant_f1_mean": (550.0, 150.0),
            "formant_f2_mean": (1500.0, 350.0),
            "formant_f3_mean": (2500.0, 400.0),
            "formant_f4_mean": (3500.0, 450.0),
            "formant_dispersion": (1000.0, 150.0),
            "vocal_tract_length_cm": (16.5, 2.0),
            "hnr_mean_db": (18.0, 6.0),
            "hnr_std_db": (4.0, 2.0),
            "cpp_mean_db": (12.0, 4.0),
            "cpp_std_db": (2.5, 1.0),
            "zcr_mean": (0.10, 0.05),
            "energy_entropy": (3.5, 0.8),
            "spectral_centroid_mean": (1800.0, 600.0),
            "spectral_centroid_std": (600.0, 250.0),
            "spectral_spread_mean": (1200.0, 300.0),
            "spectral_skewness_mean": (1.5, 0.8),
            "spectral_kurtosis_mean": (4.5, 2.0),
            "spectral_flux_mean": (0.05, 0.03),
            "spectral_flux_std": (0.03, 0.02),
            "spectral_rolloff_mean": (3200.0, 800.0),
            "spectral_flatness_mean": (0.05, 0.04),
            "spectral_crest_factor_mean": (5.0, 2.0),
            "spectral_hf_energy_ratio": (0.15, 0.08),
            "has_artificial_cutoff": (0.0, 0.5),
            "cutoff_frequency_hz": (7000.0, 1500.0),
            "prosody_speaking_rate": (4.2, 1.2),
            "prosody_articulation_rate": (4.8, 1.1),
            "prosody_syllable_count": (15.0, 8.0),
            "prosody_pause_count": (2.0, 1.5),
            "prosody_mean_pause_sec": (0.35, 0.20),
            "prosody_npvi": (55.0, 15.0),
            "prosody_rpvi": (65.0, 20.0),
            "prosody_f0_range_semitones": (6.5, 2.5),
            "prosody_pitch_slope_variance": (1200.0, 600.0),
            "prosody_is_monotone": (0.0, 0.3),
            "prosody_direction_changes": (8.0, 4.0),
            "artifact_checkerboard_ratio": (0.015, 0.02),
            "artifact_periodic_detected": (0.0, 0.3),
            "artifact_harmonic_smearing": (0.15, 0.15),
            "artifact_splice_points": (0.0, 1.0),
            "artifact_max_energy_jump_db": (4.0, 3.0),
            "artifact_max_phase_jump_rad": (0.4, 0.3),
            "phase_if_mean_deviation": (0.08, 0.04),
            "phase_if_variance": (0.005, 0.003),
            "phase_if_clustering": (0.70, 0.18),
            "phase_mgd_peak_prominence": (0.60, 0.20),
            "phase_dispersion_entropy": (0.50, 0.18),
            "phase_roughness": (1.2, 0.6),
        }

        for idx, name in enumerate(FEATURE_NAMES):
            if name in baseline_map:
                means[idx], stds[idx] = baseline_map[name]

        return means, stds
