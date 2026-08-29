"""
Configuration for Acoustic Analysis Subsystem.
"""

from typing import Optional, Tuple
from pydantic import BaseModel, ConfigDict, Field


class AcousticAnalysisConfig(BaseModel):
    """
    Configuration parameters for Pitch (F0), Jitter/Shimmer perturbation,
    Formant estimation (LPC), Harmonic-to-Noise Ratio (HNR), and Cepstral Peak Prominence (CPP).
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # --- Pitch (F0) Tracking ---
    pitch_fmin: float = Field(
        default=50.0,
        description="Minimum pitch frequency in Hz (covers deep male voices)."
    )
    pitch_fmax: float = Field(
        default=500.0,
        description="Maximum pitch frequency in Hz (covers high female/child voices)."
    )
    frame_length_ms: float = Field(
        default=30.0,
        description="Duration of acoustic analysis frames in milliseconds."
    )
    hop_length_ms: float = Field(
        default=10.0,
        description="Hop size between consecutive frames in milliseconds."
    )
    voicing_threshold: float = Field(
        default=0.45,
        description="Autocorrelation peak threshold to classify a frame as voiced."
    )

    # --- Formant Tracking (LPC) ---
    pre_emphasis_coeff: float = Field(
        default=0.97,
        description="Pre-emphasis filter coefficient for formant resonance sharpening."
    )
    lpc_order_offset: int = Field(
        default=2,
        description="LPC order = lpc_order_offset + (sample_rate / 1000)."
    )
    max_formant_freq_hz: float = Field(
        default=5000.0,
        description="Maximum formant frequency to track in Hz."
    )
    max_formant_bandwidth_hz: float = Field(
        default=800.0,
        description="Maximum bandwidth for a valid formant resonance in Hz."
    )

    # --- Voice Quality & Cepstrum ---
    cpp_quefrency_range_ms: Tuple[float, float] = Field(
        default=(2.0, 20.0),
        description="Quefrency search window in ms for Cepstral Peak Prominence (50-500Hz)."
    )
    hnr_min_db: float = Field(
        default=-10.0,
        description="Floor for Harmonic-to-Noise Ratio in dB."
    )
    hnr_max_db: float = Field(
        default=50.0,
        description="Ceiling for Harmonic-to-Noise Ratio in dB."
    )

    # --- Forensic Baseline Thresholds (Typical Human Ranges) ---
    # Synthetic / neural TTS voices often exhibit unnaturally low jitter (<0.2%) or high jitter (>3.0%)
    human_jitter_normal_range: Tuple[float, float] = Field(
        default=(0.2, 1.5),
        description="Normal human local jitter percentage range [min_pct, max_pct]."
    )
    human_shimmer_normal_range: Tuple[float, float] = Field(
        default=(1.0, 5.0),
        description="Normal human local shimmer percentage range [min_pct, max_pct]."
    )
    human_hnr_normal_range: Tuple[float, float] = Field(
        default=(12.0, 30.0),
        description="Normal human Harmonic-to-Noise Ratio range in dB."
    )
    human_cpp_normal_range: Tuple[float, float] = Field(
        default=(6.0, 22.0),
        description="Normal human Cepstral Peak Prominence range in dB."
    )
