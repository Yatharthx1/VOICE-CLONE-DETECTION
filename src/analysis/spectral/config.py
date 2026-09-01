"""
Configuration for Spectral Analysis Subsystem.
"""

from typing import Tuple
from pydantic import BaseModel, ConfigDict, Field


class SpectralAnalysisConfig(BaseModel):
    """
    Configuration parameters for Short-Time Fourier Transform (STFT), Mel-Spectrogram,
    Spectral Moments (Centroid, Spread, Skewness, Kurtosis), Flux, Rolloff, and High-Frequency Cutoff.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    n_fft: int = Field(default=1024, description="FFT window size in samples.")
    hop_length: int = Field(default=256, description="Hop length in samples (75% overlap).")
    n_mels: int = Field(default=80, description="Number of Mel frequency bands.")
    fmin: float = Field(default=20.0, description="Minimum frequency in Hz.")
    fmax: float = Field(default=8000.0, description="Maximum frequency in Hz (Nyquist at 16kHz).")
    
    rolloff_percentile: float = Field(default=0.85, description="Spectral rolloff energy threshold percentile.")
    hf_cutoff_threshold_db: float = Field(default=-50.0, description="Energy drop threshold to detect synthetic vocoder cutoff.")
    
    # Forensic bounds (Typical genuine speech spectral characteristics)
    human_centroid_range: Tuple[float, float] = Field(default=(500.0, 3200.0), description="Normal spectral centroid range in Hz.")
    human_rolloff_min_hz: float = Field(default=2500.0, description="Minimum expected spectral rolloff frequency in Hz.")
    human_flatness_max: float = Field(default=0.35, description="Maximum spectral flatness for harmonic speech.")
