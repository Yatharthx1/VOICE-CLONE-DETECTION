from typing import List

FEATURE_NAMES: List[str] = [
    # 1. Acoustic & Pitch Features (Does this sound like a human vocal cord vibrating?)
    "pitch_mean_f0",
    "pitch_std_f0",
    "pitch_min_f0",
    "pitch_max_f0",
    "pitch_voiced_fraction",
    "jitter_local_pct",
    "jitter_rap_pct",
    "jitter_ppq5_pct",
    "shimmer_local_pct",
    "shimmer_apq3_pct",
    "shimmer_apq5_pct",
    "shimmer_db",

    # 2. Formant & Vocal Tract Features (Biological throat geometry check)
    "formant_f1_mean",
    "formant_f2_mean",
    "formant_f3_mean",
    "formant_f4_mean",
    "formant_dispersion",
    "vocal_tract_length_cm",

    # 3. Voice Quality & Periodicity Features (Harmonics vs raw breath noise)
    "hnr_mean_db",
    "hnr_std_db",
    "cpp_mean_db",
    "cpp_std_db",
    "zcr_mean",
    "energy_entropy",

    # 4. Spectral Distribution & Moments (Centroid, spread, and energy flux)
    "spectral_centroid_mean",
    "spectral_centroid_std",
    "spectral_spread_mean",
    "spectral_skewness_mean",
    "spectral_kurtosis_mean",
    "spectral_flux_mean",
    "spectral_flux_std",
    "spectral_rolloff_mean",
    "spectral_flatness_mean",
    "spectral_crest_factor_mean",

    # 5. High-Frequency & Cutoff Features (Did an AI vocoder cut off frequencies at 4kHz?)
    "spectral_hf_energy_ratio",
    "has_artificial_cutoff",
    "cutoff_frequency_hz",

    # 6. Prosody & Rhythm Features (Speaking rate and robot monotone detector)
    "prosody_speaking_rate",
    "prosody_articulation_rate",
    "prosody_syllable_count",
    "prosody_pause_count",
    "prosody_mean_pause_sec",
    "prosody_npvi",
    "prosody_rpvi",
    "prosody_f0_range_semitones",
    "prosody_pitch_slope_variance",
    "prosody_is_monotone",
    "prosody_direction_changes",

    # 7. Synthesis & Vocoder Artifact Features (HiFi-GAN checkerboard & splicing glitches)
    "artifact_checkerboard_ratio",
    "artifact_periodic_detected",
    "artifact_harmonic_smearing",
    "artifact_splice_points",
    "artifact_max_energy_jump_db",
    "artifact_max_phase_jump_rad",

    # 8. Phase & Group Delay Features (Phase consistency and Modified Group Delay)
    "phase_if_mean_deviation",
    "phase_if_variance",
    "phase_if_clustering",
    "phase_mgd_peak_prominence",
    "phase_dispersion_entropy",
    "phase_roughness",
]

NUM_FEATURES: int = len(FEATURE_NAMES)
