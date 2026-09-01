"""
Configuration for Phase Analysis Subsystem.
"""

from pydantic import BaseModel, ConfigDict, Field


class PhaseAnalysisConfig(BaseModel):
    """
    Configuration parameters for Instantaneous Frequency (IF), Modified Group Delay (MGD),
    and Phase Inconsistency metrics.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    n_fft: int = Field(default=1024, description="FFT size for phase spectrum calculation.")
    hop_length: int = Field(default=256, description="Hop length for STFT phase unwrapping.")
    mgd_alpha: float = Field(default=0.4, description="Modified Group Delay parameter alpha.")
    mgd_gamma: float = Field(default=0.9, description="Modified Group Delay parameter gamma.")
    max_phase_dispersion_threshold: float = Field(default=0.45, description="Threshold for unnatural phase dispersion.")
