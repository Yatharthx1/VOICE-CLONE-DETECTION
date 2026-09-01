"""
Configuration for Synthesis Artifacts Analysis Subsystem.
"""

from pydantic import BaseModel, ConfigDict, Field


class SynthesisArtifactsConfig(BaseModel):
    """
    Configuration parameters for Neural Vocoder Checkerboard Artifacts (HiFi-GAN/MelGAN),
    Splice Boundary Detection, and Harmonic Smearing.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    checkerboard_fft_size: int = Field(default=512, description="FFT size for high-frequency periodicity detection.")
    splice_energy_jump_threshold_db: float = Field(default=18.0, description="Energy step threshold for splice detection.")
    phase_jump_threshold_rad: float = Field(default=2.5, description="Phase discontinuity threshold in radians.")
    smearing_bandwidth_hz: float = Field(default=300.0, description="Bandwidth for harmonic smearing evaluation.")
